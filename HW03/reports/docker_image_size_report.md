# Docker Image Size Report

| Image Type | Image Name | Size MB |
|---|---|---:|
| Naive | airbnb-serving:naive | 625.85 |
| Optimized | airbnb-serving:optimized | 280.65 |

## Result

The optimized image is smaller by **345.20 MB**, which is a **55.16%** reduction compared with the naive image.

## Explanation

The naive Docker image uses the full Python base image and copies the entire project directory into the container.

The optimized Docker image improves this by:

- using `python:3.11-slim`
- copying dependency files before source code for better Docker layer caching
- excluding notebooks, datasets, local artifacts, virtual environments, and cache folders through `.dockerignore`
- copying only the application source code needed for serving

This makes the optimized image more suitable for deployment because it is smaller, faster to transfer, and contains fewer unnecessary files.
