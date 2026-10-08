import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';
import { docsSchema } from '@astrojs/starlight/schema';

// The repository's /docs folder is the single source of truth for the docs.
export const collections = {
  docs: defineCollection({
    loader: glob({
      pattern: '**/*.md',
      base: '../docs',
      generateId: ({ entry }) =>
        `docs/${entry.replace(/\.md$/, '')}`.replace(/\/index$/, '') || 'docs',
    }),
    schema: docsSchema({
      extend: z.object({
        items: z
          .array(
            z.object({
              title: z.string(),
              status: z.enum(['shipping', 'building', 'planned', 'exploring']),
              summary: z.string(),
              why: z.string(),
              source: z.string().optional(),
              url: z.string().optional(),
            }),
          )
          .optional(),
      }),
    }),
  }),
};
