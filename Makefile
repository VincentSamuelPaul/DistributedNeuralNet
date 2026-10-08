train:
	@python3 neuralnet/train.py

build_server:
	@cd server && go build ./...

run_server:
	@cd server && go run main.go

compose-up:
	@docker compose up --build

compose-down:
	@docker compose down

compose-logs:
	@docker compose logs -f