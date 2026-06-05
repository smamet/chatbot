#!/usr/bin/env bash
# Sail-like wrapper around Docker Compose for the chatbot platform.
# Usage: ./sail [command] [args...]
#        ./sail up -d
#        ./sail chatbot tenant-create "Acme" --slug acme
#        ./sail test
#        ./sail shell

set -euo pipefail

cd "$(dirname "$0")"

if docker compose version &>/dev/null 2>&1; then
  DC=(docker compose)
elif command -v docker-compose &>/dev/null; then
  DC=(docker-compose)
else
  echo "Docker Compose is required. Install Docker Desktop or docker-compose-plugin." >&2
  exit 1
fi

if [ -z "${COMPOSE_FILE:-}" ]; then
  if [ -f docker-compose.override.yml ]; then
    export COMPOSE_FILE="docker-compose.yml:docker-compose.override.yml"
  else
    export COMPOSE_FILE="docker-compose.yml"
  fi
fi
API_SERVICE="${SAIL_SERVICE:-api}"
DB_SERVICE="${SAIL_DB_SERVICE:-db}"
WORKER_SERVICE="${SAIL_WORKER_SERVICE:-worker-automation}"

sail_help() {
  cat <<'EOF'
Chatbot Sail — Docker Compose shortcuts

  ./sail up [-d]              Start stack (db, api, worker, caddy)
  ./sail down                 Stop and remove containers
  ./sail stop | restart       Stop / restart services
  ./sail build [--no-cache]     Rebuild images
  ./sail ps                   List containers
  ./sail logs [service]       Follow logs (default: all)

  ./sail shell                Bash in API container
  ./sail mysql                MySQL CLI (chatbot database)
  ./sail chatbot <args>       Run `chatbot` CLI in API container
  ./sail migrate              alembic upgrade head
  ./sail test [pytest args]   Run tests in API container
  ./sail exec <svc> <cmd>     docker compose exec

  Dev: docker-compose.override.yml mounts ./src — no rebuild for code edits (API auto-reloads).
  Rebuild when pyproject.toml / Dockerfile change: ./sail build api

  Examples:
    ./sail up -d
    ./sail chatbot user-create admin@example.com -p 'your-password' --role admin
    ./sail chatbot tenant-create "My Bot" --slug my-bot
    ./sail chatbot sync my-bot /app/data/docs --fresh

  Environment: COMPOSE_FILE, SAIL_SERVICE (default: api)
EOF
}

# Pass through to compose when no special handling
compose() {
  "${DC[@]}" "$@"
}

case "${1:-}" in
  ""|help|-h|--help)
    sail_help
    ;;
  up|down|stop|start|restart|build|ps|pull|config)
    compose "$@"
    ;;
  logs)
    shift
    compose logs -f "$@"
    ;;
  shell|bash)
    compose exec "$API_SERVICE" bash
    ;;
  mysql)
    compose exec "$DB_SERVICE" mysql -u"${MYSQL_USER:-chatbot}" -p"${MYSQL_PASSWORD:-chatbot}" "${MYSQL_DATABASE:-chatbot}"
    ;;
  chatbot|artisan)
    shift
    if [ -t 0 ] && [ -t 1 ]; then
      compose exec -it "$API_SERVICE" chatbot "$@"
    else
      compose exec -T "$API_SERVICE" chatbot "$@"
    fi
    ;;
  migrate)
    compose exec -T "$API_SERVICE" alembic upgrade head
    ;;
  test)
    shift
    compose exec -T "$API_SERVICE" pytest "$@"
    ;;
  exec)
    shift
    compose exec "$@"
    ;;
  worker|worker-logs)
    shift
    compose logs -f "$WORKER_SERVICE" "$@"
    ;;
  *)
    compose "$@"
    ;;
esac
