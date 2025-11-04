#!/usr/bin/env node
import { readFile } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import path from 'node:path';
import process from 'node:process';

const rootDir = process.cwd();
const packageJsonPath = path.join(rootDir, 'package.json');
const healthcheckScript = path.join(rootDir, 'scripts', 'healthcheck.mjs');

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

function logHeader(message) {
  console.log(`\n=== ${message} ===`);
}

function runCommand(command, args, { env = {}, allowFailure = false } = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: rootDir,
      env: { ...process.env, ...env },
      stdio: 'inherit',
    });

    child.on('error', reject);
    child.on('exit', (code, signal) => {
      if (code === 0 || allowFailure) {
        resolve({ code: code ?? 0, signal: signal ?? null });
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

async function runHealthcheck() {
  console.log('\n→ Running healthcheck to verify repository state');
  const result = await runCommand(process.execPath, [healthcheckScript], { allowFailure: true });
  if (result.code === 0) {
    console.log('\n✔ Healthcheck passed. No further healing required.');
    return true;
  }
  console.log('\n⚠ Healthcheck is still failing. Continuing with the next phase.');
  return false;
}

async function main() {
  const pkg = await loadPackageJson();
  const scripts = pkg.scripts ?? {};

  const phases = [
    {
      title: 'Reinstall dependencies',
      steps: [
        {
          description: 'Installing dependencies',
          command: 'npm',
          args: ['install'],
        },
      ],
    },
    {
      title: 'Apply repository formatting',
      steps: scripts.format
        ? [
            {
              description: 'Running format script',
              command: 'npm',
              args: ['run', 'format'],
            },
          ]
        : [],
    },
    {
      title: 'Attempt automated lint fixes',
      steps: scripts.lint
        ? [
            {
              description: 'Running lint script',
              command: 'npm',
              args: ['run', 'lint'],
            },
          ]
        : [],
    },
  ].filter((phase) => phase.steps.length > 0);

  if (await runHealthcheck()) {
    return;
  }

  for (const phase of phases) {
    logHeader(phase.title);
    for (const step of phase.steps) {
      console.log(`\n→ ${step.description}`);
      await runCommand(step.command, step.args);
    }

    const healed = await runHealthcheck();
    if (healed) {
      return;
    }
  }

  console.error('\n✖ Self-heal steps completed but healthcheck is still failing. Manual intervention required.');
  process.exit(1);
}

main().catch((error) => {
  console.error(`\n✖ Self-heal failed: ${error.message}`);
  process.exit(1);
});
