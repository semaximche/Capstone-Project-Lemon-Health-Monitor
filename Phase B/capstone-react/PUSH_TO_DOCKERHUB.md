# How to Push to Docker Hub

This guide shows you how to manually push the Docker image to `davidkotler/lemon-health` repository.

## Prerequisites

1. Docker installed and running
2. Docker Hub account (`davidkotler`)
3. Logged in to Docker Hub

## Step-by-Step Instructions

### 1. Login to Docker Hub

```bash
docker login
```

Enter your Docker Hub username (`davidkotler`) and password when prompted.

### 2. Build the Image

Build the image with the correct tag format:

```bash
docker build -t davidkotler/lemon-health:tagname .
```

Replace `tagname` with your desired tag:
- `latest` - for the latest version
- `v1.0.0` - for version tags
- `dev` - for development builds
- Any custom tag you prefer

**Examples:**
```bash
# Build with latest tag
docker build -t davidkotler/lemon-health:latest .

# Build with version tag
docker build -t davidkotler/lemon-health:v1.0.0 .

# Build with custom tag
docker build -t davidkotler/lemon-health:dev .
```

### 3. Push the Image

Push the image to Docker Hub:

```bash
docker push davidkotler/lemon-health:tagname
```

**Examples:**
```bash
# Push latest
docker push davidkotler/lemon-health:latest

# Push version
docker push davidkotler/lemon-health:v1.0.0

# Push custom tag
docker push davidkotler/lemon-health:dev
```

### 4. Verify the Push

Check your Docker Hub repository:
- Visit: https://hub.docker.com/r/davidkotler/lemon-health
- You should see your pushed image with the tag

## Quick Commands Reference

```bash
# Complete workflow
docker login
docker build -t davidkotler/lemon-health:tagname .
docker push davidkotler/lemon-health:tagname

# Pull the image (from another machine)
docker pull davidkotler/lemon-health:tagname

# Run the pulled image
docker run -d -p 3000:3000 --name lemon-health-app davidkotler/lemon-health:tagname
```

## Automated Push via GitHub Actions

The GitHub Actions workflow (`.github/workflows/docker-build-push-dockerhub.yml`) will automatically push to `davidkotler/lemon-health` when you:

1. Push to `main` or `master` branch
2. Create a version tag (e.g., `v1.0.0`)
3. Manually trigger the workflow

**Setup for automated pushes:**

1. Go to GitHub repository → Settings → Secrets and variables → Actions
2. Add secrets:
   - `DOCKERHUB_USERNAME`: `davidkotler`
   - `DOCKERHUB_TOKEN`: Your Docker Hub access token

   To get a Docker Hub access token:
   - Go to https://hub.docker.com/settings/security
   - Click "New Access Token"
   - Give it a name and permissions (read & write)
   - Copy the token and add it as `DOCKERHUB_TOKEN` secret

3. Push to main/master branch - the workflow will automatically build and push!

## Tagging Strategy

The workflow automatically creates these tags:
- `latest` - Always points to the latest build from main/master
- `main` or `master` - Branch name tag
- `v1.0.0` - Semantic version (if you tag with `git tag v1.0.0`)
- `1.0` - Major.minor version
- `1` - Major version
- `main-abc1234` - Branch name + commit SHA

## Troubleshooting

### Authentication Error
```bash
# Re-login to Docker Hub
docker logout
docker login
```

### Permission Denied
- Make sure you're logged in as `davidkotler`
- Verify you have write access to the `davidkotler/lemon-health` repository

### Image Not Found
- Make sure you built the image with the correct tag
- Check: `docker images | grep lemon-health`

### Push Fails
- Check your internet connection
- Verify Docker Hub is accessible
- Check Docker Hub repository exists and you have permissions
