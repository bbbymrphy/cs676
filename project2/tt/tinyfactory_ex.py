
import tinytroupe
from tinytroupe.agent import TinyPerson
from tinytroupe.environment import TinyWorld, TinySocialNetwork
from tinytroupe.factory import TinyPersonFactory
from tinytroupe.extraction import ResultsReducer
import tinytroupe.control as control


factory = TinyPersonFactory("A random knowledge worker in a company providing marketing services.")


people = []
# try: 
#     with open('people.txt', 'r') as f:
#         lines = f.read().strip().split('\n\n')
#         print(lines)
#         people = lines
#     print(len(people))
# except: print("Generating new people...")


   
for i in range(2):
    person = factory.generate_person(temperature=2.0)
    print(person.minibio())
    people.append(person)

len(people)

with open('people.txt', 'w') as f:
    for person in people:
        f.write(person.minibio() + '\n\n')
    

company = TinyWorld("Some Corp Inc.", people)
company.make_everyone_accessible()
company.broadcast("Get some work done together, help each other. Provide a concise summary of what you are working on and the path to completion.")
company.run(5)



reducer = ResultsReducer()

def aux_extract_content(focus_agent: TinyPerson, source_agent:TinyPerson, target_agent:TinyPerson, kind:str, event: str, content: str, timestamp:str):

    if event == "TALK":
        author = focus_agent.name
    elif event == "CONVERSATION":
        if source_agent is None:
            author = "USER"
        else:
            author = source_agent.name
    else:
        raise ValueError(f"Unknown event: {event}")
    
    
    entry = (author, content)
    print(entry)
    return entry
    


reducer.add_reduction_rule("TALK", aux_extract_content)
reducer.add_reduction_rule("CONVERSATION", aux_extract_content)

df = reducer.reduce_agent_to_dataframe(people[0], column_names=["author", "content"])

df.to_csv("interactions.csv", index=False)