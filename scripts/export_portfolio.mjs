// Export the portfolio site's data as Markdown files Raggity can ingest.
//
//   node scripts/export_portfolio.mjs ~/hein-htet-portfolio corpus/portfolio
//
// Reads src/data/profile.ts (Node 23.6+ strips the TypeScript types itself) and writes one
// file per project and per role, so a citation names the exact project rather than "the
// dossier". It uses only the fields dossier() already shares with the site's `ask`
// endpoint, so nothing private (the phone number) reaches the corpus.

import { mkdir, rm, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const [siteDir, outDir = "corpus/portfolio"] = process.argv.slice(2);
if (!siteDir) {
  console.error("usage: node scripts/export_portfolio.mjs <portfolio-dir> [out-dir]");
  process.exit(2);
}

const data = await import(pathToFileURL(resolve(siteDir, "src/data/profile.ts")).href);
const { profile: p, interests, universes, projects, roles } = data;

const slug = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
const bullets = (items) => items.map((i) => `- ${i}`).join("\n");

const files = new Map();

files.set(
  "profile.md",
  `# ${p.name} (${p.alias})

${p.title}, based in ${p.location}.

${p.summary}

## Interests

${bullets(interests.map((i) => `${i.name}: ${i.line}`))}

## Education

${p.education.degree}, ${p.education.school}. ${p.education.period}. ${p.education.note}.

## Honors

${bullets(p.honors)}

## Contact

Email ${p.contact.email}. LinkedIn ${p.contact.linkedin}. GitHub ${p.contact.github}.
`,
);

for (const r of roles) {
  files.set(
    `experience/${slug(r.org)}.md`,
    `# ${r.title}, ${r.org}

${r.period}, ${r.place}.

${bullets(r.points)}
`,
  );
}

for (const x of projects) {
  files.set(
    `projects/${x.slug}.md`,
    `# ${x.title} (${x.year})

${x.summary}

${bullets(x.points)}

Stack: ${x.stack.join(", ")}.${x.metric ? `\nResult: ${x.metric}.` : ""}${
      x.link ? `\nLink: ${x.link.href}` : ""
    }
`,
  );
}

files.set(
  "skills.md",
  `# Skills by area

${universes.map((u) => `## ${u.name}\n\n${u.stack.join(", ")}.`).join("\n\n")}
`,
);

await rm(outDir, { recursive: true, force: true });
for (const [name, body] of files) {
  const path = join(outDir, name);
  await mkdir(join(path, ".."), { recursive: true });
  await writeFile(path, body);
}
console.log(`wrote ${files.size} files to ${outDir}`);
