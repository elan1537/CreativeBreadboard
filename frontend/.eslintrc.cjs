module.exports = {
  env: {
    browser: true,
    es2021: true,
  },
  extends: [
    "standard",
    "plugin:vue/vue3-recommended",
    "plugin:prettier/recommended",
  ],
  parserOptions: {
    ecmaVersion: "latest",
    sourceType: "module",
    ecmaFeatures: {
      jsx: true,
    },
  },
  plugins: ["vue", "prettier"],
  rules: {
    camelcase: ["error", { properties: "never" }],
  },
};
