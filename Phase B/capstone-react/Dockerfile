# Multi-stage Dockerfile for React Router SSR application

# Stage 1: Install dependencies
FROM node:20-alpine AS dependencies
WORKDIR /app
# Copy package files for better layer caching
COPY package.json package-lock.json ./
# Install all dependencies (including dev dependencies for build)
RUN npm ci --include=dev

# Stage 2: Build the application
FROM node:20-alpine AS build
WORKDIR /app
# Copy dependencies from previous stage
COPY --from=dependencies /app/node_modules ./node_modules
# Copy source code
COPY . .
# Build the application
RUN npm run build

# Stage 3: Production runtime
FROM node:20-alpine AS production
WORKDIR /app

# Install only production dependencies
COPY package.json package-lock.json ./
RUN npm ci --omit=dev && npm cache clean --force

# Copy built application from build stage
COPY --from=build /app/build ./build

# Create non-root user for security
RUN addgroup -g 1001 -S nodejs && \
    adduser -S reactuser -u 1001 && \
    chown -R reactuser:nodejs /app

USER reactuser

# Expose port (React Router default is 3000, adjust if needed)
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD node -e "require('http').get('http://localhost:3000', (r) => {process.exit(r.statusCode === 200 ? 0 : 1)})"

# Start the application
CMD ["npm", "run", "start"]