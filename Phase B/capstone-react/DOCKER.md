# Docker Setup Guide

This guide explains how to build, run, and deploy the Capstone React application using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose (optional, for easier local development)

## Building the Docker Image

### Build locally

```bash
docker build -t capstone-react:latest .
```

### Build with custom tag

```bash
docker build -t capstone-react:v1.0.0 .
```

## Running the Container

### Using Docker directly

```bash
docker run -d \
  --name capstone-react-app \
  -p 3000:3000 \
  capstone-react:latest
```

### Using Docker Compose

```bash
docker-compose up -d
```

To view logs:
```bash
docker-compose logs -f
```

To stop:
```bash
docker-compose down
```

## CI/CD Workflows

### GitHub Container Registry (ghcr.io)

The project includes a GitHub Actions workflow (`.github/workflows/docker-build-push.yml`) that automatically:

- Builds the Docker image on push to main/master/develop branches
- Pushes to GitHub Container Registry
- Tags images based on branch, version tags, and SHA

**Setup:**
1. No additional setup needed - uses `GITHUB_TOKEN` automatically
2. Images will be available at: `ghcr.io/<your-username>/<repo-name>`

**Pull the image:**
```bash
docker pull ghcr.io/<your-username>/<repo-name>:latest
```

### Docker Hub

The project includes a Docker Hub workflow (`.github/workflows/docker-build-push-dockerhub.yml`) that:

- Builds and pushes to Docker Hub
- Requires Docker Hub credentials as GitHub secrets

**Setup:**
1. Go to GitHub repository → Settings → Secrets and variables → Actions
2. Add secrets:
   - `DOCKERHUB_USERNAME`: Your Docker Hub username
   - `DOCKERHUB_TOKEN`: Your Docker Hub access token

**Pull the image:**
```bash
docker pull davidkotler/lemon-health:latest
```

**Manual push (if needed):**
```bash
# Build the image
docker build -t davidkotler/lemon-health:tagname .

# Login to Docker Hub
docker login

# Push the image
docker push davidkotler/lemon-health:tagname
```

## Image Tags

The workflows automatically create tags:
- `latest` - Latest build from main/master branch
- `<branch-name>` - Builds from specific branches
- `<branch-name>-<sha>` - Specific commit SHA
- `v1.0.0` - Semantic version tags
- `1.0` - Major.minor version
- `1` - Major version

## Multi-Architecture Support

The workflows build for both:
- `linux/amd64` (Intel/AMD 64-bit)
- `linux/arm64` (ARM 64-bit, Apple Silicon, etc.)

## Environment Variables

If you need to pass environment variables to the container:

```bash
docker run -d \
  --name capstone-react-app \
  -p 3000:3000 \
  -e NODE_ENV=production \
  capstone-react:latest
```

## Health Check

The container includes a health check that verifies the application is responding on port 3000.

Check health status:
```bash
docker ps
# Look for "healthy" status
```

## Troubleshooting

### View logs
```bash
docker logs capstone-react-app
```

### Access container shell
```bash
docker exec -it capstone-react-app sh
```

### Rebuild without cache
```bash
docker build --no-cache -t capstone-react:latest .
```

### Remove old images
```bash
docker image prune -a
```

## Production Deployment

For production deployment, consider:

1. **Use specific version tags** instead of `latest`
2. **Set up proper reverse proxy** (nginx, traefik, etc.)
3. **Configure environment variables** for API endpoints
4. **Set up monitoring** and logging
5. **Use orchestration** (Kubernetes, Docker Swarm) for scaling

## Security Notes

- The Dockerfile runs as a non-root user (`reactuser`)
- Only production dependencies are included in the final image
- Health checks are configured for monitoring
- Multi-stage build reduces image size
