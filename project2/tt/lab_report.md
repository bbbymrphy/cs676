# Lab Report: TinyTroupe Agent Simulation and Social Interaction Analysis

**Course:** CS676
**Date:** November 1, 2025
**Experiment:** Multi-Agent Social Simulation using TinyTroupe Framework

---

## 1. Introduction

### 1.1 Objective
This experiment explores the capabilities of the TinyTroupe framework for simulating realistic social interactions between AI-powered agents in a workplace environment. The primary objectives were to:
- Generate diverse synthetic personas using `TinyPersonFactory`
- Simulate workplace interactions in a shared environment (`TinyWorld`)
- Extract and analyze conversation data for insights into agent behavior
- Evaluate the framework's ability to maintain character consistency and generate meaningful dialogue

### 1.2 Background
TinyTroupe is a multi-agent simulation framework that enables the creation of AI-powered personas capable of engaging in realistic social interactions. This experiment focuses on simulating knowledge workers in a marketing services company to observe emergent collaborative behaviors.

---

## 2. Methodology

### 2.1 Experimental Setup

The experiment was implemented in [tinyfactory_ex.py](tinyfactory_ex.py) using the following components:

**Agent Generation:**
- Factory specification: "A random knowledge worker in a company providing marketing services"
- Number of agents: 2
- Temperature parameter: 2.0 (high variability for diverse personas)

**Environment Configuration:**
- World name: "Some Corp Inc."
- Interaction mode: Full accessibility (all agents can communicate)
- Simulation rounds: 5

**Data Collection:**
- Results extraction using `ResultsReducer`
- Focus events: "TALK" and "CONVERSATION"
- Output format: CSV with author and content columns

### 2.2 Code Structure

The experiment follows this workflow:

1. **Persona Generation** ([tinyfactory_ex.py:10-27](tinyfactory_ex.py#L10-L27))
   ```python
   factory = TinyPersonFactory("A random knowledge worker...")
   for i in range(2):
       person = factory.generate_person(temperature=2.0)
       people.append(person)
   ```

2. **Environment Setup** ([tinyfactory_ex.py:36-39](tinyfactory_ex.py#L36-L39))
   ```python
   company = TinyWorld("Some Corp Inc.", people)
   company.make_everyone_accessible()
   company.broadcast("Get some work done together...")
   company.run(5)
   ```

3. **Data Extraction** ([tinyfactory_ex.py:43-67](tinyfactory_ex.py#L43-L67))
   - Custom reduction function to extract author-content pairs
   - Conversion to pandas DataFrame for analysis

---

## 3. Results

### 3.1 Generated Personas

The factory generated two distinct personas with rich background details:

**Agent 1: Eleanor Mitchell**
- Age: 42 years old
- Occupation: Senior Librarian
- Location: Portland, Oregon, USA
- Personality Traits: Thoughtful, empathetic, values deep connections
- Interests: Local history, literature, environmental sustainability
- Hobbies: Calligraphy, gardening, cello playing
- Professional Focus: Community engagement, equitable access to information

**Agent 2: Harold Benson**
- Age: 52 years old
- Occupation: Senior Maintenance Technician
- Location: Columbus, Ohio, USA
- Personality Traits: Dependable, pragmatic, honest, reserved
- Interests: Classic American muscle cars, fishing, woodworking
- Professional Focus: Technical expertise, mentoring young tradespeople

### 3.2 Interaction Analysis

The simulation generated 9 conversational exchanges captured in [interactions.csv](interactions.csv):

**Observation 1: Character Consistency**
Despite the factory specification requesting "knowledge workers in a marketing company," the generated agents maintained their original professional identities (librarian and maintenance technician) throughout the conversation. This demonstrates strong character consistency but reveals a potential limitation in factory specification adherence.

**Observation 2: Professional Domain Mapping**
- Eleanor discussed library work: organizing local history collections, grant writing, educational programs
- Harold discussed maintenance work: hydraulic press inspection, valve testing, equipment repair

**Observation 3: Social Dynamics**
The agents demonstrated:
- **Cross-domain empathy**: Eleanor acknowledged Harold's hands-on technical work
- **Analogy formation**: Both recognized parallels between maintaining machines and maintaining information systems
- **Mutual respect**: Exchange of appreciation for each other's professional contributions
- **Conversational coherence**: Logical flow from initial prompt to mutual understanding

**Observation 4: Interaction Depth**
The conversation evolved through distinct phases:
1. Initial task sharing (responses to broadcast)
2. Acknowledgment and comparison
3. Deepening mutual understanding
4. Expression of respect and closure

---

## 4. Technical Implementation Details

### 4.1 Data Persistence
The code implements two persistence mechanisms:
- **people.txt**: Stores minibios of generated personas (commented caching logic at [tinyfactory_ex.py:14-20](tinyfactory_ex.py#L14-L20))
- **interactions.csv**: Stores conversation history with author attribution

### 4.2 Event Processing
The custom extraction function ([tinyfactory_ex.py:45-60](tinyfactory_ex.py#L45-L60)) differentiates between:
- **TALK events**: Authored by the focus agent
- **CONVERSATION events**: Attributed to source agent or "USER" for broadcasts
- Handles edge cases (None source agents)

### 4.3 Reduction Pipeline
The `ResultsReducer` applies custom rules to:
- Filter relevant interaction types
- Extract structured data (author, content tuples)
- Transform to tabular format for analysis

---

## 5. Discussion

### 5.1 Strengths
1. **Rich Persona Generation**: The factory created detailed, believable characters with comprehensive backgrounds
2. **Coherent Dialogue**: Conversations maintained logical flow and context awareness
3. **Social Intelligence**: Agents demonstrated empathy, active listening, and relationship building
4. **Extensibility**: The reduction framework enables flexible data extraction strategies

### 5.2 Limitations
1. **Factory Specification Adherence**: Generated personas did not match the "marketing services" domain specification
2. **Limited Interaction Complexity**: The 5-round simulation produced relatively shallow conversations
3. **No Task Completion**: Agents discussed work but did not demonstrate collaborative task execution
4. **Persona Diversity**: With only 2 agents, network effects and group dynamics cannot be observed

### 5.3 Potential Improvements
1. Increase simulation rounds for deeper interactions
2. Add specific task objectives beyond generic "get work done" prompts
3. Implement persona validation to ensure factory specification compliance
4. Expand agent count to study emergent group behaviors
5. Add quantitative metrics (sentiment analysis, topic modeling, turn-taking patterns)

---

## 6. Conclusions

This experiment successfully demonstrated the TinyTroupe framework's capability to:
- Generate diverse synthetic personas with rich psychological profiles
- Simulate realistic workplace conversations with coherent dialogue
- Extract structured interaction data for analysis

The key finding is that TinyTroupe agents exhibit strong character consistency and social intelligence, engaging in contextually appropriate conversations that demonstrate empathy and mutual understanding. However, the experiment also revealed challenges in controlling persona generation to match specific domain requirements.

Future work should focus on:
1. Refining factory specifications for better domain alignment
2. Implementing longer-term simulations to study relationship development
3. Adding quantitative evaluation metrics for interaction quality
4. Exploring multi-agent collaboration on concrete tasks

---

## 7. Appendices

### Appendix A: Output Files
- **people.txt**: Generated persona minibios
- **interactions.csv**: Complete conversation transcript with 9 exchanges

### Appendix B: Dependencies
- tinytroupe (agent simulation framework)
- pandas (data processing, implicitly through ResultsReducer)

### Appendix C: Key Parameters
- Temperature: 2.0 (high diversity)
- Simulation rounds: 5
- Agent count: 2
- Broadcast message: "Get some work done together, help each other. Provide a concise summary of what you are working on and the path to completion."

---

## References

1. TinyTroupe Framework Documentation
2. Source code: [tinyfactory_ex.py](tinyfactory_ex.py)
3. Generated data: [people.txt](people.txt), [interactions.csv](interactions.csv)
