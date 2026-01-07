/** @type {import('tailwindcss').Config} */
import defaultTheme from 'tailwindcss/defaultTheme';

export default {
	content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
	theme: {
		extend: {
            fontFamily: {
                // Uses clean system fonts (San Francisco on Mac)
                sans: ['Inter', 'system-ui', 'sans-serif'],
            }
        },
	},
	plugins: [],
};