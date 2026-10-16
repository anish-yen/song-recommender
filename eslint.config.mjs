export default [
  { ignores: ['.next/**', 'node_modules/**', 'generated/**'] },
  {
    files: ['src/**/*.js', 'tests/**/*.js'],
    languageOptions: { ecmaVersion: 'latest', sourceType: 'module', parserOptions: { ecmaFeatures: { jsx: true } } },
    rules: { 'no-unused-vars': 'off' },
  },
];
