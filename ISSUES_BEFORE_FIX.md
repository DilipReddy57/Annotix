# Issues Report & Remediation Plan

## Executive Summary
A comprehensive audit of the codebase revealed **235 issues** across the backend and frontend. These range from minor stylistic linting errors to critical performance bottlenecks and potential application crashes.

## Backend Analysis
**Tool**: `ruff`
**Total Issues**: 193
**Criticality**: High

### Key Findings:
1.  **Latency / Performance**:
    - `backend/api/routes/tasks.py`: Synchronous file I/O (`shutil.copyfileobj`) inside an `async` route. This blocks the main event loop, causing the entire API to hang during large file uploads.
    - `backend/api/routes/tasks.py`: `get_analytics` loads all annotations into memory to calculate stats. This is O(N) memory usage and will crash with large datasets.
2.  **Code Quality / Potential Bugs**:
    - `backend/sam3/sam3/train/loss/sigmoid_focal_loss.py`: Use of `lambda` for kernel grid definitions.
    - Multiple scripts use bare `except:` clauses, swallowing `KeyboardInterrupt` and making debugging impossible.
    - 130+ unused imports (`F401`) cluttering the code.

## Frontend Analysis
**Tool**: `eslint` (with `typescript-eslint`)
**Total Issues**: 42
**Criticality**: Medium-High

### Key Findings:
1.  **React Violations (Critical)**:
    - `frontend/src/context/AuthContext.tsx`: Updating state synchronously inside `useEffect`. This can cause cascading renders and performance issues.
    - `frontend/src/components/ui/annotation-rain.tsx`: Calling impure functions (`Math.random()`) during the render phase. This leads to unstable UI and hydration mismatches.
2.  **Hooks & Stability**:
    - Missing dependencies in `useEffect` hooks in `SpotlightHero` and `ProjectView`.
3.  **Type Safety**:
    - Widespread use of `any` type, defeating TypeScript's safety guarantees.

## Remediation Plan
I will proceed to fix these issues in the following order:
1.  **Auto-Fix**: Apply automated linting fixes to backend.
2.  **Backend Criticals**: Rewrite blocking I/O and fix bare exceptions.
3.  **Frontend Criticals**: Refactor React components to follow Hooks rules.
4.  **Polish**: Manually address remaining linting errors.
