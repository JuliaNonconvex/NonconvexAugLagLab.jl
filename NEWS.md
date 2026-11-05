# NEWS

## 0.1.4

- Fixed invalid stdlib compat bounds (`LinearAlgebra = "1.12.0 - 1"`) that
  accidentally shipped in 0.1.3 and made the package uninstallable on
  Julia 1.10/1.11 (users silently resolved to 0.1.2).
- Requires Julia 1.10+.
