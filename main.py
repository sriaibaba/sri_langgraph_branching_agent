"""Interactive LangGraph agent. Type exit to quit."""

from agent import ask_agent


def main() -> None:
    while True:
        query = input("Hi User Ask Question: ")
        if query.strip().lower() == "exit":
            break
        if not query.strip():
            continue
        print(ask_agent(query))


if __name__ == "__main__":
    main()
