---- initial attempt at running the Lisa example ----

        tesbobbymurphy@bobbys-MacBook-Pro tinytroupe % python3 main.py

        !!!!
        DISCLAIMER: TinyTroupe relies on Artificial Intelligence (AI) models to generate content. 
        The AI models are not perfect and may produce inappropriate or inacurate results. 
        For any serious or consequential use, please review the generated content before using it.
        !!!!

        Looking for default config on: /Users/bobbymurphy/cs676/project2/tinytroupe/tinytroupe/utils/../config.ini
        Failed to find custom config on: /Users/bobbymurphy/cs676/project2/tinytroupe/config.ini
        Will use only default values. IF THINGS FAIL, TRY CUSTOMIZING MODEL, API TYPE, etc.
        TinyTroupe version: 0.5.2
        Current date and time (local): 2025-10-30 23:01:40
        Current date and time (UTC):   2025-10-31 03:01:40

        =================================
        Current TinyTroupe configuration 
        =================================
        [OpenAI]
        api_type = openai
        azure_api_version = 2023-05-15
        model = gpt-4.1-mini
        reasoning_model = o3-mini
        embedding_model = text-embedding-3-small
        max_tokens = 32000
        temperature = 1.5
        freq_penalty = 0.1
        presence_penalty = 0.1
        timeout = 480
        max_attempts = 5
        waiting_time = 1
        exponential_backoff_factor = 5
        reasoning_effort = high
        cache_api_calls = False
        cache_file_name = openai_api_cache.pickle
        max_content_display_length = 4000

        [Simulation]
        parallel_agent_generation = True
        parallel_agent_actions = True
        rai_harmful_content_prevention = True
        rai_copyright_infringement_prevention = True

        [Cognition]
        enable_memory_consolidation = True
        min_episode_length = 15
        max_episode_length = 50
        episodic_memory_fixed_prefix_length = 10
        episodic_memory_lookback_length = 20

        [ActionGenerator]
        max_attempts = 2
        enable_quality_checks = False
        enable_regeneration = True
        enable_direct_correction = False
        enable_quality_check_for_persona_adherence = True
        enable_quality_check_for_selfconsistency = False
        enable_quality_check_for_fluency = False
        enable_quality_check_for_suitability = False
        enable_quality_check_for_similarity = False
        continue_on_failure = True
        quality_threshold = 5

        [Logging]
        loglevel = ERROR

        USER --> Lisa Carter: [CONVERSATION] 
                > Talk to Oscar to know more about him
        ─────────────────────────────────────────────────────────── Chat Room step 1 of 4 ────────────────────────────────────────────────────────────
        Oscar acts: [THINK] 
                > I have no current stimuli or tasks to address. I should consider what to do next or
                > wait for new input.
        Lisa Carter acts: [REACH_OUT] 

        Oscar acts: [DONE] 

        Lisa Carter acts: [TALK] 
                        > Hi Oscar, I wanted to take a moment to get to know you better. Could you tell me
                        > a bit about your role and interests?
        Lisa Carter acts: [DONE] 

        Chat Room --> Lisa Carter: [SOCIAL] 
                    > Oscar was successfully reached out, and is now available for interaction.
        Chat Room --> Oscar: [SOCIAL] 
                    > Lisa Carter reached out to you, and is now available for interaction.
        Lisa Carter --> Oscar: [CONVERSATION] 
                        > Hi Oscar, I wanted to take a moment to get to know you better. Could you tell me
                        > a bit about your role and interests?
        ─────────────────────────────────────────────────────────── Chat Room step 2 of 4 ────────────────────────────────────────────────────────────
        Oscar acts: [THINK] 
                > Lisa Carter has reached out and is now available for interaction. She wants to know
                > about my role and interests. I should respond warmly and professionally, sharing about
                > my work as an architect at Awesome Inc., focusing on sustainable modular housing, and
                > mention my interests in modernist architecture, sustainability, and playing guitar to
                > give a personal touch.
        Lisa Carter acts: [TALK] 
                        > Hi Oscar, I was just about to ask you about your role and interests. Could you
                        > share a bit about yourself?
        Lisa Carter acts: [DONE] 

        Oscar acts: [TALK] 
                > Hi Lisa! I'm Oscar, an architect at Awesome Inc. I specialize in creating standard
                > elements for new apartment buildings, focusing on sustainable and functional designs
                > that are also aesthetically pleasing. Outside of work, I have a passion for modernist
                > architecture, sustainable practices, and I enjoy playing the guitar to unwind. What
                > about you?
        Oscar acts: [DONE] 

        Lisa Carter --> Oscar: [CONVERSATION] 
                        > Hi Oscar, I was just about to ask you about your role and interests. Could you
                        > share a bit about yourself?
        Oscar --> Lisa Carter: [CONVERSATION] 
                > Hi Lisa! I'm Oscar, an architect at Awesome Inc. I specialize in creating standard
                > elements for new apartment buildings, focusing on sustainable and functional designs
                > that are also aesthetically pleasing. Outside of work, I have a passion for modernist
                > architecture, sustainable practices, and I enjoy playing the guitar to unwind. What
                > about you?
        ─────────────────────────────────────────────────────────── Chat Room step 3 of 4 ────────────────────────────────────────────────────────────
        Oscar acts: [THINK] 
                > Lisa is asking again about my role and interests. I already prepared a response
                > previously, so I should repeat it with warmth and professionalism, maybe adding a bit
                > more detail to keep it engaging.
        Lisa Carter acts: [THINK] 
                        > Oscar has shared that he is an architect specializing in sustainable and
                        > functional designs, with interests in modernist architecture and playing the
                        > guitar. I should respond by sharing a bit about myself, my role as a data
                        > scientist at Microsoft, and my interests in AI and machine learning, to build
                        > rapport and find common ground.
        Oscar acts: [TALK] 
                > Hi Lisa! As I mentioned, I'm an architect at Awesome Inc., where I focus on designing
                > standard elements for new apartment buildings. My goal is to create sustainable,
                > functional, and aesthetically pleasing designs that make it easier for customers to
                > choose their apartment configurations. Outside of work, I have a strong interest in
                > modernist architecture, sustainable practices, and I enjoy playing the guitar to relax.
                > What about you?
        Lisa Carter acts: [TALK] 
                        > Hi Oscar! It's great to meet you. I'm Lisa, a data scientist at Microsoft working
                        > on the M365 Search team. I focus on improving search relevance using machine
                        > learning and natural language processing. I also have a passion for AI and enjoy
                        > cooking and playing the piano in my free time. It sounds like we both appreciate
                        > sustainability and creativity in our work and hobbies!
        Oscar acts: [DONE] 

        Lisa Carter acts: [DONE] 

        Lisa Carter --> Oscar: [CONVERSATION] 
                        > Hi Oscar! It's great to meet you. I'm Lisa, a data scientist at Microsoft working
                        > on the M365 Search team. I focus on improving search relevance using machine
                        > learning and natural language processing. I also have a passion for AI and enjoy
                        > cooking and playing the piano in my free time. It sounds like we both appreciate
                        > sustainability and creativity in our work and hobbies!
        Oscar --> Lisa Carter: [CONVERSATION] 
                > Hi Lisa! As I mentioned, I'm an architect at Awesome Inc., where I focus on designing
                > standard elements for new apartment buildings. My goal is to create sustainable,
                > functional, and aesthetically pleasing designs that make it easier for customers to
                > choose their apartment configurations. Outside of work, I have a strong interest in
                > modernist architecture, sustainable practices, and I enjoy playing the guitar to relax.
                > What about you?
        ─────────────────────────────────────────────────────────── Chat Room step 4 of 4 ────────────────────────────────────────────────────────────
        Oscar acts: [THINK] 
                > Lisa shared about her role as a data scientist at Microsoft, working on search
                > relevance with machine learning and NLP. She also enjoys cooking and playing piano, and
                > values sustainability and creativity like me. I should respond warmly, acknowledge the
                > similarities, and maybe ask about her work or hobbies to keep the conversation flowing.
        Lisa Carter acts: [TALK] 
                        > Hi Oscar! Thanks for sharing more about your work and interests. I really admire
                        > your focus on sustainability and functionality in architecture. As for me, I'm a
                        > data scientist at Microsoft on the M365 Search team, where I work on improving
                        > search relevance using machine learning and natural language processing. Outside
                        > of work, I enjoy cooking, experimenting with new recipes, and playing the piano
                        > to relax. It sounds like we both appreciate creativity and sustainability in our
                        > own ways!
        Oscar acts: [TALK] 
                > Hi Lisa! It's great to meet you too. Your work on improving search relevance with
                > machine learning sounds fascinating, and I really admire how AI is shaping the future.
                > I also love that we both value sustainability and creativity—it definitely makes our
                > work more meaningful. How did you get interested in data science and AI? And what's
                > your favorite dish to cook when you have some free time?
        Lisa Carter acts: [DONE] 

        Oscar acts: [DONE] 

        Lisa Carter --> Oscar: [CONVERSATION] 
                        > Hi Oscar! Thanks for sharing more about your work and interests. I really admire
                        > your focus on sustainability and functionality in architecture. As for me, I'm a
                        > data scientist at Microsoft on the M365 Search team, where I work on improving
                        > search relevance using machine learning and natural language processing. Outside
                        > of work, I enjoy cooking, experimenting with new recipes, and playing the piano
                        > to relax. It sounds like we both appreciate creativity and sustainability in our
                        > own ways!
        Oscar --> Lisa Carter: [CONVERSATION] 
                > Hi Lisa! It's great to meet you too. Your work on improving search relevance with
                > machine learning sounds fascinating, and I really admire how AI is shaping the future.
                > I also love that we both value sustainability and creativity—it definitely makes our
                > work more meaningful. How did you get interested in data science and AI? And what's
                > your favorite dish to cook when you have some free time?


----  ----



