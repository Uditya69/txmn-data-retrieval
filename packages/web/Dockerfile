FROM node:22-slim AS build
WORKDIR /app
COPY packages/web/package.json packages/web/package-lock.json ./
RUN npm ci
COPY packages/web .
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY packages/web/docker-entrypoint.sh /docker-entrypoint.d/40-env-config.sh
RUN chmod +x /docker-entrypoint.d/40-env-config.sh
EXPOSE 80
