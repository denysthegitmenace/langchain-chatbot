# langchain-chatbot

To use, you should have:

- all the python packages from `requirements.txt` installed
- the environment variable `OPENAI_API_KEY` set with your OpenAI API key.

# Setting up a local Postgres Db

Sample database DDls are in `Chinook_PostgreSql.sql`

Running Postgres db locally in a Docker container:

`docker pull postgres`

`docker run --name my_postgres_container -e POSTGRES_PASSWORD=password -d -p 5432:5432 postgres -c log_statement=all`

Creating sample db in Postgres container

`docker cp ./Chinook_PostgreSql.sql my_postgres_container:/docker-entrypoint-initdb.d/Chinook_PostgreSql.sql`

`docker exec -u postgres my_postgres_container psql postgres postgres -f docker-entrypoint-initdb.d/Chinook_PostgreSql.sql`

# References

Langchain fuzzy chain reference --> https://api.python.langchain.com/en/latest/_modules/langchain/chains/openai_functions/citation_fuzzy_match.html#create_citation_fuzzy_match_chain
OpenAI API reference --> https://platform.openai.com/docs/api-reference/chat/create
Playground --> https://platform.openai.com/playground/?model=gpt-3.5-turbo-16k
Citations --> https://platform.openai.com/docs/guides/gpt-best-practices/strategy-provide-reference-text
