// @ts-check
// The shared max-models docs setup (@max-models/starlight-theme): served at
// https://max-models.github.io/wellbalanced/, with math, the API reference and a link check.
import { defineDocs } from '@max-models/starlight-theme';

export default defineDocs({
	repo: 'wellbalanced',
	title: 'wellbalanced',
	description: 'Global (0D) and 1D fluid models of low-temperature plasmas.',
	pythonPackage: 'wellbalanced',
	// the method cards: a Numerics section with the overview (numerics/index.mdx) and one page per
	// module, in the order of the model, and a Numerics section on each method's API page
	methodcard: {
		pages: true,
		index: false,
		labels: {
			'wellbalanced.reactions': 'Reactions',
			'wellbalanced.global_model': 'Global model',
			'wellbalanced.cr': 'OpenCR rates',
			'wellbalanced.neutralizer': 'Neutralizer',
			'wellbalanced.driver': 'RF driver (0D)',
			'wellbalanced.magnetron': 'Magnetron',
			'wellbalanced.fluid1d': 'Fluid 1D',
			'wellbalanced.spatial.grid': 'Cell grid',
			'wellbalanced.spatial.fluid': 'Fluid 2D',
			'wellbalanced.spatial.inductive': 'Inductive field 2D',
			'wellbalanced.spatial.edges': 'Edge grid',
			'wellbalanced.spatial.cutcell': 'Cut cells',
			'wellbalanced.spatial.inductive3d': 'Inductive field 3D',
			'wellbalanced.spatial.iterative': 'Iterative solver',
			'wellbalanced.spatial.coupling': 'Field and plasma',
			'wellbalanced.spatial.neutrals': 'Neutral gas 2D',
			'wellbalanced.driftfluid2d': 'Drift fluid 2D',
			'wellbalanced.dispersion': 'Local dispersion',
			'wellbalanced.gasdynamics': 'Gas dynamics',
			'wellbalanced.transport': 'Transport budget',
		},
	},
	nav: [
		{ label: 'Getting started', link: '/getting-started/installation/' },
		{ label: 'Tutorials', link: '/tutorials/' },
		{ label: 'Full examples', link: '/full-examples/' },
		{ label: 'Numerics', link: '/numerics/' },
		{ label: 'Development', link: '/development/docs/' },
		{ label: 'API', link: '/api/wellbalanced/' },
	],
	sidebar: [
		{
			label: 'Getting started',
			items: [
				{ label: 'Installation', slug: 'getting-started/installation' },
				{ label: 'Quickstart', slug: 'getting-started/quickstart' },
			],
		},
		{
			label: 'Tutorials',
			items: [{ autogenerate: { directory: 'tutorials' } }],
		},
		{
			label: 'Full examples',
			items: [{ autogenerate: { directory: 'full-examples' } }],
		},
		{
			label: 'Development',
			items: [
				{ label: 'Building the docs', slug: 'development/docs' },
				{ label: 'Publishing', slug: 'development/publishing' },
				{ label: '3D field discretisation', slug: 'development/field-3d' },
			],
		},
	],
});
