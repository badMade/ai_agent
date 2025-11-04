#!/usr/bin/env node
import { readFile } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import path from 'node:path';
import process from 'node:process';

const rootDir = process.cwd();
const packageJsonPath = path.join(rootDir, 'package.json');

async function loadPackageJson() {
  try {
    const raw = await readFile(packageJsonPath, 'utf8');
    return JSON.parse(raw);
  } catch (error) {
    if (error && error.code === 'ENOENT') {
      throw new Error('package.json not found. Run this script from the repository root.');
    }
    throw error;
  }
}

function logStep(message) {
  console.log(`\n▶ ${message}`);
}

function runCommand(command, args, { env = {} } = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: rootDir,
      env: { ...process.env, ...env },
      stdio: 'inherit',
    });

    child.on('error', reject);
    child.on('exit', (code, signal) => {
      if (code === 0) {
        resolve();
        return;
      }
      if (signal) {
        reject(new Error(`Command ${command} terminated with signal ${signal}`));
        return;
      }
      reject(new Error(`Command ${command} exited with code ${code ?? 1}`));
    });
  });
}

async function runIfScript(pkg, scriptName, options = {}) {
  const scripts = pkg.scripts ?? {};
  if (!scripts[scriptName]) {
    console.log(`Skipping ${scriptName} (script not defined).`);
    return;
  }
  logStep(`Running ${scriptName}`);
  await runCommand('npm', ['run', scriptName], options);
}

async function runWorkspaceBuilds(pkg) {
  const workspaces = pkg.workspaces;
  if (!workspaces || (Array.isArray(workspaces) && workspaces.length === 0)) {
    console.log('No workspaces defined; skipping workspace smoke builds.');
    return;
  }
  logStep('Running workspace build smoke tests');
  await runCommand('npm', ['run', 'build', '--workspaces', '--if-present']);
}

async function main() {
  const pkg = await loadPackageJson();
  logStep('Starting repository healthcheck');

  await runIfScript(pkg, 'typecheck');
  await runIfScript(pkg, 'lint');
  await runIfScript(pkg, 'test', { env: { CI: '1' } });
  await runWorkspaceBuilds(pkg);

  console.log('\n✔ Healthcheck completed successfully.');
}

main().catch((error) => {
  console.error(`\n✖ Healthcheck failed: ${error.message}`);
  process.exit(1);
});
