// Rewrites relative links between the repository's markdown files so the same
// files read correctly on GitHub and on the site.
import path from 'node:path';

const REPO = 'https://github.com/sachncs/context/blob/master/';

export function docsLinks({ base }) {
  return () => (tree, file) => {
    const source = file.path ?? '';
    const marker = `${path.sep}docs${path.sep}`;
    const index = source.lastIndexOf(marker);
    if (index === -1) return;
    const fromDocs = source.slice(index + marker.length);
    const dir = path.posix.dirname(fromDocs.split(path.sep).join('/'));
    const visit = (node) => {
      if (node.type === 'link' && typeof node.url === 'string') {
        const url = node.url;
        if (!/^([a-z]+:|#|\/)/i.test(url)) {
          const [target, hash = ''] = url.split('#');
          const resolved = path.posix.normalize(path.posix.join(dir, target));
          if (resolved.startsWith('..')) {
            const repoPath = path.posix.normalize(path.posix.join('docs', dir, target));
            node.url = REPO + repoPath + (hash ? `#${hash}` : '');
          } else if (target.endsWith('.md')) {
            const route = resolved.replace(/\.md$/, '').replace(/(^|\/)index$/, '');
            node.url = `${base}docs/${route}${route ? '/' : ''}`.replace(/\/{2,}/g, '/') + (hash ? `#${hash}` : '');
          }
        }
      }
      if (node.children) node.children.forEach(visit);
    };
    visit(tree);
  };
}
