from graph_db import get_driver

driver = get_driver()

with driver.session() as session:
    result = session.run("RETURN 'Connection successful!' AS message")
    for record in result:
        print(record["message"])