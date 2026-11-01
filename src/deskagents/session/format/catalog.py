"""Upstream: packages/session/session-format/src/catalog.ts"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from .chain import SessionFormatMigrationStream, create_session_format_chain
from .context import SessionFormatEventCollector
from .errors import SessionFormatError, SessionFormatUnsupportedMigrationError
from .json import (
    inspect_session_format_version,
    session_format_version,
    snapshot_session_format_header,
)
from .types import (
    SessionFormatArtifact,
    SessionFormatArtifactDecoder,
    SessionFormatCatalogOptions,
    SessionFormatCodec,
    SessionFormatEvent,
    SessionFormatEventRun,
    SessionFormatHeader,
    SessionFormatHeaderReadResult,
    SessionFormatJsonObject,
    SessionFormatMigrationContext,
    SessionFormatRestoreOptions,
)

_Restorer = Callable[[SessionFormatArtifact], SessionFormatArtifact]


def _reason(error: BaseException) -> str:
    return error.args[0] if len(error.args) == 1 and isinstance(error.args[0], str) else str(error)


def create_session_format_catalog(options: SessionFormatCatalogOptions) -> SessionFormatCatalog:
    """Compile a build-static physical codec and adjacent migration catalog."""
    return SessionFormatCatalog(options)


class SessionFormatCatalog:
    """Build-static physical dispatch and adjacent migration catalog."""

    def __init__(self, options: SessionFormatCatalogOptions) -> None:
        self._chain = create_session_format_chain(options)
        self._options = options
        self.current_version = self._chain.current_version
        codecs: dict[int, SessionFormatCodec] = {}
        for codec in options.codecs:
            version = session_format_version(codec.version, "Session format codec version")
            if version in codecs:
                raise SessionFormatError(f"Session format codec v{version} is duplicated")
            codecs[version] = codec
        for version in range(self.current_version + 1):
            if version not in codecs:
                raise SessionFormatError(f"Session format codec v{version} is missing")
        if len(codecs) != self.current_version + 1:
            invalid = next(v for v in codecs if v > self.current_version)
            raise SessionFormatError(
                f"Session format codec v{invalid} is newer than current v{self.current_version}"
            )
        self._codecs = codecs

    def read_header(self, header_value: object) -> SessionFormatHeaderReadResult:
        """Classify and translate one header without reading event rows."""
        try:
            stored = inspect_session_format_version(header_value)
        except Exception as error:
            return _malformed(self.current_version, error)
        if stored > self.current_version:
            return SessionFormatHeaderReadResult(
                status="unsupported",
                stored_version=stored,
                target_version=self.current_version,
                reason=f"stored Session uses newer format v{stored}; "
                f"this build writes v{self.current_version}",
            )
        codec = self._codecs[stored]
        try:
            decoded = snapshot_session_format_header(
                codec.decode_header(header_value), f"format v{stored} header"
            )
            header = self._chain.migrate_header(decoded)
        except SessionFormatUnsupportedMigrationError as error:
            return SessionFormatHeaderReadResult(
                status="unsupported",
                stored_version=stored,
                target_version=self.current_version,
                reason=_reason(error),
            )
        except Exception as error:
            return _malformed(self.current_version, error, stored)
        return SessionFormatHeaderReadResult(
            status="current" if stored == self.current_version else "migration-required",
            stored_version=stored,
            target_version=self.current_version,
            header=header,
        )

    def _artifact_codec(self, header_value: object) -> tuple[int, SessionFormatCodec]:
        stored = inspect_session_format_version(header_value)
        if stored > self.current_version:
            raise SessionFormatUnsupportedMigrationError(
                f"stored Session uses newer format v{stored}; "
                f"this build writes v{self.current_version}"
            )
        return stored, self._codecs[stored]

    def encode_current_header(
        self, header: SessionFormatHeader, inherited_event_count: int
    ) -> SessionFormatJsonObject:
        """Encode one current physical header record."""
        if inspect_session_format_version(header) != self.current_version:
            raise SessionFormatError(
                f"encodeCurrent requires Session format v{self.current_version}"
            )
        encoded = self._options.current_encoder.encode_header(header, inherited_event_count)
        if inspect_session_format_version(encoded) != self.current_version:
            raise SessionFormatError("current Session codec returned a non-current header")
        return encoded

    def encode_current_event(self, event: SessionFormatEvent) -> SessionFormatJsonObject:
        """Encode one current physical event record."""
        return self._options.current_encoder.encode_event(event)

    def create_restore(
        self, header_value: object, options: SessionFormatRestoreOptions
    ) -> SessionFormatRestore:
        """Create one single-pass physical-row restore into current logical events."""
        stored, codec = self._artifact_codec(header_value)
        decoder = codec.create_decoder(header_value, options.recovery)
        source_cut = getattr(decoder, "header_inherited_event_count", None)
        if stored == self.current_version:
            return _CurrentRestore(
                decoder,
                source_cut,
                self._options.restore_current if options.validation == "current" else _identity,
                self.current_version,
            )
        collector = SessionFormatEventCollector()
        migration = self._chain.create_stream(decoder.header, source_cut, collector)
        return _MigratingRestore(
            decoder,
            source_cut,
            migration,
            collector,
            self._options.restore_current
            if options.validation == "current"
            else self._options.restore_transformed_current,
            options.validation,
            stored,
            self.current_version,
        )


class SessionFormatRestore:
    """One caller-owned physical-row restore whose final value is a current artifact."""

    header: SessionFormatHeader
    """Current logical header available before body decoding."""

    def decode_row(self, row_value: object) -> None:
        """Decode one physical row in file order."""
        raise NotImplementedError

    def finish(self) -> SessionFormatArtifact:
        """Finish every decoder and migration stage and return the current artifact."""
        raise NotImplementedError


class _CurrentRestore(SessionFormatRestore):
    def __init__(
        self,
        decoder: SessionFormatArtifactDecoder,
        source_cut: int | None,
        restore: _Restorer,
        current_version: int,
    ) -> None:
        self._decoder = decoder
        self._source_cut = source_cut
        self._restore = restore
        self._current_version = current_version
        self._collector = SessionFormatEventCollector()
        self.header = decoder.header

    def decode_row(self, row_value: object) -> None:
        self._decoder.decode_row(row_value, self._collector)

    def finish(self) -> SessionFormatArtifact:
        cut = _finish_decoder(self._decoder, self._collector, self._source_cut)
        artifact = SessionFormatArtifact(self.header, cut, self._collector.values)
        return _restore_current_version(self._restore(artifact), self._current_version)


class _MigratingRestore(SessionFormatRestore):
    def __init__(
        self,
        decoder: SessionFormatArtifactDecoder,
        source_cut: int | None,
        migration: SessionFormatMigrationStream,
        collector: SessionFormatEventCollector,
        restore: _Restorer,
        validation: Literal["transformed", "current"],
        source_version: int,
        current_version: int,
    ) -> None:
        self._decoder = decoder
        self._source_cut = source_cut
        self._migration = migration
        self._collector = collector
        self._restore = restore
        self._validation = validation
        self._source_version = source_version
        self._current_version = current_version
        self.header = migration.header

    def decode_row(self, row_value: object) -> None:
        self._decoder.decode_row(row_value, self)

    def emit_event(self, event: SessionFormatEvent) -> None:
        self._migration.emit_event(event)

    def emit_run(self, run: SessionFormatEventRun) -> None:
        self._migration.emit_run(run)

    def finish(self) -> SessionFormatArtifact:
        _finish_decoder(self._decoder, self, self._source_cut)
        artifact = SessionFormatArtifact(
            self.header, self._migration.finish(), self._collector.values
        )
        try:
            restored = self._restore(artifact)
        except Exception as error:
            if self._validation == "current" or isinstance(
                error, SessionFormatUnsupportedMigrationError
            ):
                raise
            raise SessionFormatUnsupportedMigrationError(
                f"Session migration from v{self._source_version} to v{self._current_version} "
                f"refuses the transformed artifact: {_reason(error)}"
            ) from error
        return _restore_current_version(restored, self._current_version)


def _finish_decoder(
    decoder: SessionFormatArtifactDecoder,
    context: SessionFormatMigrationContext,
    source_cut: int | None,
) -> int:
    cut = decoder.finish(context)
    if source_cut is not None and cut != source_cut:
        raise SessionFormatError("streaming decoder changed its predeclared inherited cut")
    return cut


def _restore_current_version(
    artifact: SessionFormatArtifact, current_version: int
) -> SessionFormatArtifact:
    if artifact.header["version"] != current_version:
        raise SessionFormatError(
            f"current Session restorer returned v{artifact.header['version']}; "
            f"expected v{current_version}"
        )
    return artifact


def _identity(artifact: SessionFormatArtifact) -> SessionFormatArtifact:
    return artifact


def _malformed(
    target_version: int, error: BaseException, stored_version: int | None = None
) -> SessionFormatHeaderReadResult:
    return SessionFormatHeaderReadResult(
        status="malformed",
        stored_version=stored_version,
        target_version=target_version,
        reason=_reason(error),
    )
