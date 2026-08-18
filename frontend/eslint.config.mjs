import coreWebVitals from "eslint-config-next/core-web-vitals";
import typescript from "eslint-config-next/typescript";

// eslint-config-next 16 ships flat configs directly. Going through
// @eslint/eslintrc's FlatCompat shim instead throws "Converting circular
// structure to JSON" on ESLint 9, because the shim tries to serialise the
// plugin graph while validating it.
const eslintConfig = [
  ...coreWebVitals,
  ...typescript,
  { ignores: [".next/**", "out/**", "node_modules/**"] },
];

export default eslintConfig;
