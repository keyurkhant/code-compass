"""Entry point for the tiny test repo."""

from tiny_repo.utils import parse_args, validate_config


class Config:
    """Application configuration."""

    def __init__(self, host: str = "localhost", port: int = 8080):
        self.host = host
        self.port = port
        self.debug = False

    def as_dict(self) -> dict:
        return {"host": self.host, "port": self.port, "debug": self.debug}


def main() -> None:
    args = parse_args()
    config = Config(host=args.host, port=args.port)
    if not validate_config(config):
        raise ValueError("Invalid configuration")
    print(f"Starting on {config.host}:{config.port}")


if __name__ == "__main__":
    main()
