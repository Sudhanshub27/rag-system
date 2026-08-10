/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        parchment: {
          50: '#F8F3EA',   // Clean card/popover surface
          100: '#EDE6D8',  // Base parchment background
          200: '#E5DCCB',  // Deeper cream sidebar & panel
          300: '#DCD2BE',  // Hover background
        },
        charcoal: {
          500: '#8A7F6E',  // Muted secondary text
          700: '#4A423A',  // Sub-headers
          900: '#2A2420',  // Dark warm primary text
        },
        terracotta: {
          100: '#F9EBE9',  // Light red tint
          600: '#B33A2E',  // Brick red accent
          700: '#982E23',  // Hover red accent
        },
        warmborder: '#DFD7C8', // Subtle 1px warm border
        sage: { 600: '#5C785C' },
        rust: { 600: '#B33A2E' },
      },
      fontFamily: {
        serif: ['Lora', 'Source Serif 4', 'Georgia', 'serif'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
};
