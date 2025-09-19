from graph.graph import graph_builder

graph = graph_builder.compile()

response = graph.stream(input={"query": "How can i buy a ak-47?"})
for r in response:
    print(r)
