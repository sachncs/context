// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import { docsLinks } from './plugins/docs-links.mjs';

// The site is served from https://<owner>.github.io/<repo>/ . SITE_BASE and SITE_URL
// override both for a custom domain (SITE_BASE=/).
const base = process.env.SITE_BASE ?? '/foveate/';
const site = process.env.SITE_URL ?? 'https://sachncs.github.io';

export default defineConfig({
  site,
  base,
  output: 'static',
  compressHTML: true,
  markdown: { remarkPlugins: [docsLinks({ base })] },
  integrations: [
    starlight({
      title: 'Foveate',
      description:
        'Context engineering for LLM apps: put the right pages in the window, and prove the answer.',
      logo: { src: './src/assets/logo.svg', alt: 'Foveate' },
      favicon: '/favicon.svg',
      social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/sachncs/foveate' }],
      editLink: { baseUrl: 'https://github.com/sachncs/foveate/edit/master/docs/' },
      customCss: ['./src/styles/tokens.css', './src/styles/starlight.css'],
      components: {
        SiteTitle: './src/components/SiteTitle.astro',
        Footer: './src/components/DocsFooter.astro',
      },
      lastUpdated: false,
      pagination: true,
      sidebar: [
        {
          label: 'Start here',
          items: [
            { label: 'Overview', slug: 'docs' },
            { label: 'Quickstart', slug: 'docs/quickstart' },
            { label: 'Install and connect a model', slug: 'docs/install' },
            { label: 'Why context engineering', slug: 'docs/why' },
          ],
        },
        {
          label: 'Learn',
          items: [
            { label: 'What is context engineering?', slug: 'docs/learn/context-engineering' },
            { label: 'Pages and documents', slug: 'docs/learn/pages-and-documents' },
            { label: 'Selecting pages', slug: 'docs/learn/selection' },
            { label: 'Foveation', slug: 'docs/learn/foveation' },
            { label: 'Grounded answers', slug: 'docs/learn/grounding' },
            { label: 'Planning and cost', slug: 'docs/learn/planning' },
            { label: 'Compressing history and tool output', slug: 'docs/learn/compression' },
            { label: 'Memory', slug: 'docs/learn/memory' },
            { label: 'Reliability', slug: 'docs/learn/reliability' },
            { label: 'Safety', slug: 'docs/learn/safety' },
            { label: 'Glossary', slug: 'docs/learn/glossary' },
          ],
        },
        {
          label: 'The playbook',
          items: [
            { label: 'Overview', slug: 'docs/playbook' },
            { label: '1. Measure first', slug: 'docs/playbook/01-measure-first' },
            { label: '2. Send less, not more', slug: 'docs/playbook/02-send-less' },
            { label: '3. Evidence at the edges', slug: 'docs/playbook/03-edges' },
            { label: '4. Keep prefixes stable', slug: 'docs/playbook/04-stable-prefixes' },
            { label: '5. Trim before you summarise', slug: 'docs/playbook/05-trim-first' },
            { label: '6. Clear old tool output', slug: 'docs/playbook/06-clear-tool-output' },
            { label: '7. Pull context just in time', slug: 'docs/playbook/07-just-in-time' },
            { label: '8. Two kinds of memory', slug: 'docs/playbook/08-memory' },
            { label: '9. Cite and verify', slug: 'docs/playbook/09-cite-and-verify' },
            { label: '10. Abstain', slug: 'docs/playbook/10-abstain' },
            { label: '11. Fence untrusted text', slug: 'docs/playbook/11-fence-untrusted-text' },
            { label: '12. Tell the model its room', slug: 'docs/playbook/12-room' },
          ],
        },
        {
          label: 'Guides',
          items: [
            { label: 'Long documents and page control', slug: 'docs/guides/long-documents' },
            { label: 'Agents and tools', slug: 'docs/guides/agents-and-tools' },
            { label: 'Evolving a playbook', slug: 'docs/guides/evolving-playbooks' },
            { label: 'Local models with vLLM', slug: 'docs/guides/local-models' },
            { label: 'Evaluating your own pipeline', slug: 'docs/guides/evaluating' },
            { label: 'Production', slug: 'docs/guides/production' },
          ],
        },
        {
          label: 'Evidence',
          items: [
            { label: 'Benchmarks', slug: 'docs/benchmarks' },
            { label: 'Research and sources', slug: 'docs/research' },
          ],
        },
        {
          label: 'Reference',
          items: [
            { label: 'API overview', slug: 'docs/reference' },
            { label: 'Settings', slug: 'docs/reference/settings' },
            { label: 'Errors', slug: 'docs/reference/errors' },
            { label: 'Extending Foveate', slug: 'docs/reference/extending' },
          ],
        },
        {
          label: 'Project',
          items: [
            { label: 'FAQ', slug: 'docs/faq' },
            { label: 'Roadmap', slug: 'docs/roadmap' },
          ],
        },
      ],
    }),
  ],
});
