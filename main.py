from graph.graph import graph


def main():
    print("SupportDoc Agent — Microsoft Intune troubleshooting assistant")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Your question: ").strip()
        if not query or query.lower() == "exit":
            break

        result = graph.invoke({"query": query, "retrieval_count": 0})
        print(f"\n{result['generation']}\n")


if __name__ == "__main__":
    main()

