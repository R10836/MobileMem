import { spawnSync } from "node:child_process";
import { readdirSync } from "node:fs";
import path from "node:path";

const targets = [
  { directory: "assets/web/js", extension: ".js" },
  { directory: "scripts", extension: ".mjs" },
];

const files = targets
  .flatMap(({ directory, extension }) =>
    readdirSync(directory, { withFileTypes: true })
      .filter((entry) => entry.isFile() && path.extname(entry.name) === extension)
      .map((entry) => path.join(directory, entry.name)),
  )
  .sort();

for (const file of files) {
  const result = spawnSync(process.execPath, ["--check", file], { stdio: "inherit" });
  if (result.status !== 0) process.exit(result.status ?? 1);
}

console.log(`Syntax check passed (${files.length} files).`);
