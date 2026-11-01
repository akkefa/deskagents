/** Inventory literal upstream cases with the upstream TypeScript parser.
 * Usage: node scripts/upstream_case_inventory.mjs /path/to/deepseek-harness
 * Unmatched cases stay deferred-review; this does not assert applicability or parity.
 */
import { createRequire } from 'node:module';
import { readFileSync, writeFileSync, readdirSync } from 'node:fs';
import { resolve, relative, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const upstream = resolve(process.argv[2]);
const require = createRequire(join(upstream, 'package.json'));
const ts = require('typescript');
const map = JSON.parse(readFileSync(join(root, 'docs/upstream.json')));
const commit = execFileSync('git', ['-C', upstream, 'rev-parse', 'HEAD'], {encoding:'utf8'}).trim();
if (commit !== map.upstream.commit) throw new Error('upstream checkout must match pin');
const catalog = JSON.parse(readFileSync(join(root, 'docs/compatibility-cases.json')));
const known = new Map(catalog.cases.map(row => [row.source, row]));
const groups = ['core/agent-loop','core/agent','core/tools','core/system-prompt','llm/llm','llm/llm-deepseek'];
function files(path) {
  return readdirSync(path, {withFileTypes:true}).flatMap(entry => entry.isDirectory() ? files(join(path,entry.name)) : entry.name.endsWith('.spec.ts') ? [join(path,entry.name)] : []);
}
function base(expression) {
  if (ts.isIdentifier(expression)) return expression.text;
  if (ts.isCallExpression(expression)) return base(expression.expression);
  if (ts.isPropertyAccessExpression(expression)) return base(expression.expression);
  return '';
}
const specs=[];
for (const group of groups) for (const path of files(join(upstream,'packages',group,'tests')).sort()) {
  const name=relative(upstream,path).split('\\').join('/');
  const source=ts.createSourceFile(name,readFileSync(path,'utf8'),ts.ScriptTarget.Latest,true);
  const cases=[];
  function visit(node, parents=[]) {
    if (ts.isCallExpression(node)) {
      const kind=base(node.expression), title=node.arguments[0];
      const callback=node.arguments.find(a => ts.isArrowFunction(a)||ts.isFunctionExpression(a));
      if (callback && title && (ts.isStringLiteral(title)||ts.isNoSubstitutionTemplateLiteral(title)) && ['describe','it','test'].includes(kind)) {
        const full=[...parents,title.text];
        if (kind==='describe') { visit(callback,full); return; }
        const origin=name+'::'+full.join(' > '), mapped=known.get(origin);
        cases.push({source:origin,classification:mapped?.classification ?? 'deferred-review',
          python_tests:mapped?.python_tests ?? [],
          notes:mapped?.notes || (mapped ? '' : 'Not yet source-indexed; applicability and porting remain deferred for the alpha. Parameterized titles represent a family of cases.')});
        return;
      }
    }
    ts.forEachChild(node, child=>visit(child,parents));
  }
  visit(source);
  specs.push({path:name,cases});
}
const output={upstream_commit:commit,scope:groups,method:'TypeScript AST literal it/test titles; parameterized families are not expanded. Dynamic titles and custom test wrappers may be absent. Deferred-review is not a claim of non-applicability.',specs};
writeFileSync(join(root,'docs/upstream-case-inventory.json'),JSON.stringify(output,null,2)+'\n');
console.log(`${specs.length} spec files, ${specs.reduce((n,s)=>n+s.cases.length,0)} literal case families inventoried`);
