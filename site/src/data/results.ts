import fs from 'node:fs';
import path from 'node:path';

const root = path.resolve(process.cwd(), 'src', 'data', 'results');

export function readSummary(...parts: string[]): any | null {
  const file = path.join(root, ...parts, 'summary.json');
  return fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, 'utf-8')) : null;
}

export const pipelineNames: Record<string, string> = {
  'full-context': 'Send everything',
  truncate: 'Truncate',
  'naive-rag': 'Plain RAG',
  foveate: 'Foveate',
};
