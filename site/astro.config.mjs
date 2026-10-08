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
      social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/sachncs/context' }],
      editLink: { baseUrl: 'https://github.com/sachncs/context/edit/master/docs/' },
      customCss: ['./src/styles/tokens.css', './src/styles/starlight.css'],
      components: { SiteTitle: './src/components/SiteTitle.astro' },
      lastUpdated: false,
      pagination: true,
      sidebar: [
        {
          label: 'Start here',
          items: [
            { label: 'Overview', slug: 'docs' },
            { label: 'Why context engineering', slug: 'docs/why' },
            { label: 'Quickstart', slug: 'docs/quickstart' },
            { label: 'Concepts', slug: 'docs/concepts' },
          ],
        },
        {
          label: 'Guides',
          items: [
            { label: 'Long documents and page control', slug: 'docs/guides/long-documents' },
            { label: 'Agents and tools', slug: 'docs/guides/agents-and-tools' },
            { label: 'Local models with vLLM', slug: 'docs/guides/local-models' },
            { label: 'Evaluating your own pipeline', slug: 'docs/guides/evaluating' },
            { label: 'Production', slug: 'docs/guides/production' },
          ],
        },
        {
          label: 'Evidence',
          items: [
            { label: 'Benchmarks', slug: 'docs/benchmarks' },
            { label: 'The twelve rules', slug: 'docs/playbook' },
            { label: 'Research and sources', slug: 'docs/research' },
          ],
        },
        { label: 'Reference', items: [{ label: 'API and settings', slug: 'docs/reference' }] },
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
