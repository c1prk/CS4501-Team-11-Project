import { useState, useEffect, useCallback, useRef } from "react";

// ── DATA ──────────────────────────────────────────────────────
const TOPICS = [
  {
    id: "social-cognition",
    title: "Social Cognition",
    session: "09",
    icon: "🧠",
    color: "#6366f1",
    concepts: [
      {
        term: "Naïve Psychology",
        definition: "Children's commonsense understanding of other people's behaviors, thoughts, and feelings. It's their informal 'theory' about why people do what they do.",
        example: "A 2-year-old understands that if Mommy wants a cookie, she'll go get one."
      },
      {
        term: "Theory of Mind (ToM)",
        definition: "Understanding that other people have mental states — beliefs, desires, intentions — that drive their actions, and that these may differ from your own.",
        example: "Knowing that your friend thinks there's candy in the box even though you saw it was actually pencils."
      },
      {
        term: "False Belief Understanding",
        definition: "The ability to recognize that someone can hold a belief that is incorrect. Considered THE critical test of Theory of Mind.",
        example: "Sally-Anne task: Sally puts marble in basket, leaves. Anne moves it to box. Where will Sally look? Correct = basket (her false belief). 3-year-olds fail; 4-5 year-olds pass."
      },
      {
        term: "False Content Task",
        definition: "Show a familiar container (e.g., Smarties box), reveal unexpected contents (pencils), then ask what someone else would think is inside. Tests understanding that others can have false beliefs about contents.",
        example: "3-year-olds say 'pencils' (fail — they can't separate own knowledge from others'). 5-year-olds correctly say 'Smarties.'"
      },
      {
        term: "Theory of Mind Module (TOMM)",
        definition: "A proposed INNATE brain mechanism specifically devoted to understanding other people. This is the NATIVIST view of Theory of Mind.",
        example: "Nativists argue the TOMM is why even infants show early signs of understanding others' goals and intentions."
      },
      {
        term: "Nativism vs. Empiricism (ToM)",
        definition: "Nativists: Innate mechanisms (TOMM) provide basic understanding; nurture develops it further. Empiricists: General learning mechanisms (perceiving, associating, remembering) + executive function development enable ToM.",
        example: "Nativist evidence: Very young infants understand goals. Empiricist evidence: Executive function (inhibiting own perspective) correlates with passing false belief tasks."
      },
      {
        term: "Understanding Intentions",
        definition: "Infants understand that others' behavior is goal-directed. By 6 months they get this; by 14 months they show 'rational imitation' — they copy the INTENT, not the exact action.",
        example: "Gergely et al. (2002): Adult turns on light with forehead (hands full). 14-month-olds use their hands instead — they understood the GOAL, not just the movement."
      },
      {
        term: "Understanding Desires",
        definition: "By 18 months, children understand that others can want different things than they do. By 2 years, they connect desires to actions.",
        example: "Broccoli study (Repacholi & Gopnik, 1999): 18-month-olds give the experimenter broccoli (which SHE liked) even though THEY preferred crackers."
      },
      {
        term: "Implicit vs. Explicit False Belief",
        definition: "Toddlers show implicit understanding (looking behavior) before they can explicitly pass verbal false belief tasks. Three explanations: (1) One system — tasks too hard, (2) Pattern detection without mental state understanding, (3) Two systems — implicit early, explicit later.",
        example: "Onishi & Baillargeon (2005): 15-month-olds looked longer when an actor searched in the 'wrong' place relative to her false belief, suggesting implicit understanding."
      },
      {
        term: "Sticky Mittens",
        definition: "Velcro mittens that help even 3-month-olds grasp objects. After this experience, infants better understand others' reaching goals — linking own goal-directed action to understanding others'.",
        example: "Sommerville et al. (2005): 3-month-olds with sticky mitten experience understood others' goal-directed reaching, while those without did not."
      }
    ],
    quiz: [
      {
        q: "Three-year-old Maya watches as her mom puts cookies in a jar, then leaves. Maya's dad moves the cookies to a cupboard. When asked where Mom will look for cookies, Maya says 'the cupboard.' This demonstrates Maya's failure to understand:",
        options: ["Conservation", "Object permanence", "False belief", "Accommodation"],
        correct: 2,
        explanation: "Maya fails the false belief task — she can't separate her own knowledge (cookies are in cupboard) from her mom's belief (cookies are in jar). Typical for 3-year-olds."
      },
      {
        q: "A researcher shows 14-month-olds an adult who turns on a lamp using her forehead because her hands are occupied holding a blanket. When given the chance, most infants turn the light on with their hands instead. This is BEST explained by:",
        options: ["Classical conditioning — infants associate the light with the forehead", "Rational imitation — infants understood the goal and chose an efficient means", "Assimilation — infants fit the action into existing schemas", "Perceptual narrowing — infants can't perceive the forehead movement"],
        correct: 1,
        explanation: "This is Gergely et al.'s (2002) rational imitation study. Infants understood the adult's GOAL (turn on light) and that the forehead method was constrained (hands full), so they used the more efficient method."
      },
      {
        q: "Which statement best represents the EMPIRICIST position on Theory of Mind?",
        options: ["Children are born with a brain module specifically for understanding others (TOMM)", "General cognitive abilities like executive function and learning from experience underlie ToM development", "False belief understanding is present from birth but hidden by task demands", "Theory of Mind develops entirely through parental instruction"],
        correct: 1,
        explanation: "Empiricists argue ToM arises from general learning mechanisms — perceiving, associating, remembering — plus developing executive function (like inhibiting your own perspective)."
      },
      {
        q: "In the broccoli/crackers study (Repacholi & Gopnik), 18-month-olds gave the experimenter broccoli when she expressed liking it, even though the toddlers preferred crackers. This demonstrates understanding of:",
        options: ["False beliefs", "Others' desires can differ from one's own", "Conservation of quantity", "Scaffolding"],
        correct: 1,
        explanation: "By 18 months, children have a more solid understanding that other people can want different things than they do — they gave the experimenter what SHE wanted, not what they wanted."
      }
    ]
  },
  {
    id: "social-dev-1",
    title: "Theories I: Psychoanalytic & Learning",
    session: "10",
    icon: "📚",
    color: "#ec4899",
    concepts: [
      {
        term: "Freud's Id",
        definition: "The most primitive personality structure, present from BIRTH. Entirely unconscious. Operates on the PLEASURE PRINCIPLE — seeks immediate gratification. Visible in selfish, impulsive behavior throughout life.",
        example: "A hungry infant screaming for food right now is pure id — no patience, no reasoning, just 'I WANT IT NOW.'"
      },
      {
        term: "Freud's Ego",
        definition: "Emerges at END OF FIRST YEAR. Operates on the REALITY PRINCIPLE — reason and good sense. Resolves conflicts between the id's demands and the external world. Develops into sense of self.",
        example: "A toddler who waits briefly for a snack instead of screaming — the ego mediates between wanting it now (id) and understanding they need to wait."
      },
      {
        term: "Freud's Superego",
        definition: "Emerges around AGES 3-6. The CONSCIENCE — internalized parental rules and standards for acceptable behavior. Helps child avoid actions that would lead to guilt. Develops through identification with same-sex parent.",
        example: "A 5-year-old who tells a lie, then feels guilty and confesses — the superego's internalized rules create guilt that motivates confession."
      },
      {
        term: "Erikson's Psychosocial Theory",
        definition: "Built on Freud but emphasized EGO and IDENTITY. Incorporated social factors. Eight life stages (infancy to old age), each with a CRISIS to resolve. Fixed order. Know first 5 stages.",
        example: "Unlike Freud's focus on sexual drives, Erikson focused on social challenges at each age — trust, autonomy, initiative, industry, identity."
      },
      {
        term: "Erikson Stage 1: Trust vs. Mistrust (0-1 yr)",
        definition: "Crucial issue: Learning to trust caregivers through consistent, warm care. If trust doesn't develop now, person will have difficulty forming intimate relationships later.",
        example: "An infant whose cries are consistently responded to learns 'the world is safe and reliable.'"
      },
      {
        term: "Erikson Stage 2: Autonomy vs. Shame/Doubt (1-3 yrs)",
        definition: "Crucial issue: Achieving a strong sense of autonomy while adjusting to social demands. Supportive atmosphere allows self-control without loss of self-esteem. Severe punishment → doubt about own abilities.",
        example: "A toddler insisting 'I do it myself!' when getting dressed — striving for independence."
      },
      {
        term: "Erikson Stage 3: Initiative vs. Guilt (4-6 yrs)",
        definition: "Crucial issue: Taking initiative toward goals while developing conscience. Children identify with parents, internalize standards, set high goals. Conflicts with others → guilt. Not overly controlling parenting is key.",
        example: "A 5-year-old who decides to organize a game but feels guilty when other kids don't want to play their way."
      },
      {
        term: "Erikson Stage 4: Industry vs. Inferiority (6-puberty)",
        definition: "Crucial issue: Mastering cognitive and social skills; learning to work industriously and cooperate with peers. Success → competence; failure → excessive feelings of inadequacy.",
        example: "A child who feels proud after completing a difficult school project vs. one who feels 'I'm terrible at everything' after struggles."
      },
      {
        term: "Erikson Stage 5: Identity vs. Role Confusion (Adolescence)",
        definition: "Crucial issue: Achieving a core sense of identity. Shift from child to adult — must resolve questions about who they are and their future roles. Without resolution → role confusion.",
        example: "A teenager exploring different friend groups, career interests, and values to figure out 'who am I really?'"
      },
      {
        term: "Classical Conditioning (Watson)",
        definition: "Learning by ASSOCIATING an initial stimulus with one that always evokes a reflexive response. The conditioned stimulus eventually evokes the response on its own. Watson believed conditioning is the PRIMARY mechanism of development.",
        example: "Little Albert: White rat (neutral) paired with loud noise (unconditioned stimulus) → rat alone produced fear (conditioned response). Also: child fears doctor's white coat after painful injections."
      },
      {
        term: "Operant Conditioning (Skinner)",
        definition: "Learning the relation between one's own BEHAVIOR and its CONSEQUENCES. Reinforcement increases future behavior likelihood. Punishment decreases it.",
        example: "Reinforcement: Shaking a rattle → interesting sound → shakes more. Punishment: Reaching toward hot stove → 'NO!' → reaches less."
      },
      {
        term: "Social Learning Theory (Bandura)",
        definition: "Children learn by OBSERVING and IMITATING others, not just through direct reinforcement. Imitation is not just mimicry. Key concepts: vicarious reinforcement (learning from watching others' consequences) and reciprocal determinism (child, behavior, and environment all influence each other bidirectionally).",
        example: "Bobo doll study: Children who watched an adult act aggressively toward a doll imitated the aggression — without any direct reinforcement for doing so."
      },
      {
        term: "Reciprocal Determinism",
        definition: "Bandura's concept that the child is ACTIVE in their own development — behavior, personal factors (cognition/temperament), and environment all influence each other in both directions. Distinguishes social learning from pure behaviorism.",
        example: "An aggressive child (personal factor) picks fights (behavior) which leads peers to avoid them (environment) which increases anger (personal) → a cycle."
      }
    ],
    quiz: [
      {
        q: "A 5-year-old tells a lie, then later feels guilty and confesses. How would FREUD explain the guilt?",
        options: ["The id is seeking pleasure through confession", "The ego is balancing reality demands", "The superego's internalized rules create guilt for violating standards", "Classical conditioning paired lying with punishment"],
        correct: 2,
        explanation: "The superego (emerging ages 3-6) is the conscience — internalized parental rules. When the child violates these standards, the superego produces guilt, motivating confession."
      },
      {
        q: "A researcher observes that children who watch a peer get praised for sharing begin sharing more themselves, even though THEY were never directly reinforced. This best illustrates:",
        options: ["Classical conditioning", "Operant conditioning", "Vicarious reinforcement (social learning)", "Erikson's initiative vs. guilt stage"],
        correct: 2,
        explanation: "Vicarious reinforcement (Bandura) = learning from watching the consequences of OTHERS' behavior. No direct reinforcement needed — just observation."
      },
      {
        q: "Which is a key difference between Freud's and Erikson's theories?",
        options: ["Freud emphasized stages; Erikson did not", "Erikson focused more on ego/identity and social factors; Freud focused on instinctual drives", "Freud included adulthood; Erikson only covered childhood", "Erikson rejected the importance of early experiences"],
        correct: 1,
        explanation: "Erikson built on Freud but shifted emphasis from sexual drives to ego/identity development and incorporated social/cultural factors. Both used stages, and Erikson actually extended them to cover the full lifespan."
      },
      {
        q: "Gender-typical behavior: a girl plays with dolls and a boy plays with trucks. How would a BEHAVIORIST explain this differently from a social learning theorist?",
        options: ["Behaviorist: direct reinforcement/punishment shaped the preference. Social learning: the child observed and imitated same-gender models", "Both would explain it the same way — through reinforcement", "Behaviorist: innate biology. Social learning: environment", "Behaviorist: observation. Social learning: reinforcement"],
        correct: 0,
        explanation: "Behaviorists focus on direct consequences — girls praised for dolls, boys praised for trucks. Social learning theorists add observational learning — children watch same-gender peers/adults and imitate, even without direct reinforcement."
      },
      {
        q: "An infant learns that crying leads to being picked up and comforted. Over time, the infant cries more frequently when wanting attention. This is an example of:",
        options: ["Classical conditioning", "Operant conditioning (reinforcement)", "Vicarious learning", "Assimilation"],
        correct: 1,
        explanation: "The infant's behavior (crying) is reinforced by its consequence (being picked up/comforted), increasing the future likelihood of that behavior. This is operant conditioning."
      }
    ]
  },
  {
    id: "social-dev-2",
    title: "Theories II: Social Cognitive & Ecological",
    session: "11",
    icon: "🌍",
    color: "#14b8a6",
    concepts: [
      {
        term: "Social Cognitive Theories (Overview)",
        definition: "Child ACTIVELY processes social information — attends, infers, interprets. Cognition is critical to social development. Key concept: SELF-SOCIALIZATION — the child actively shapes their own development through how they think about the social world.",
        example: "Unlike learning theories which focus on external reinforcement, social cognitive theories focus on what happens INSIDE the child's mind when they encounter social situations."
      },
      {
        term: "Selman's Role-Taking",
        definition: "The ability to take another's point of view — to adopt another person's perspective. Develops in stages from egocentric (can't take other perspectives) → recognizing perspectives differ → stepping into another's shoes → taking a 'generalized' group perspective.",
        example: "Young child: 'Everyone likes pizza because I like pizza.' Older child: 'My friend might not like pizza even though I do, and society generally values diverse food preferences.'"
      },
      {
        term: "Dodge's Hostile Attributional Bias",
        definition: "The tendency to assume others have HOSTILE INTENTIONS when a situation is AMBIGUOUS. Children with this bias expect antagonism, react with retaliation → becomes a self-fulfilling prophecy. Predicted by early HARSH PARENTING. Effects persist into adulthood.",
        example: "A child is bumped in the hallway (ambiguous — could be accident). A child with hostile attributional bias thinks 'they did that on PURPOSE to hurt me' and pushes back → starts a fight."
      },
      {
        term: "Bronfenbrenner's Bioecological Model",
        definition: "Multiple NESTED levels of environmental influence simultaneously affect development. The child is at the center, and each system represents increasingly distant influences. Also emphasizes the child's ACTIVE role in selecting and influencing contexts.",
        example: "A child's development is shaped by family (micro), parent-teacher relationship (meso), parent's workplace stress (exo), cultural values about education (macro), and changes over time (chrono)."
      },
      {
        term: "Microsystem",
        definition: "The IMMEDIATE environment and relationships the child directly participates in. Activities, roles, and relationships in the child's direct experience.",
        example: "Family, school classroom, peer group, childcare, religious institution — places where the child actually IS and interacts."
      },
      {
        term: "Mesosystem",
        definition: "CONNECTIONS between microsystems — how different parts of the child's immediate world interact with each other.",
        example: "Parent-teacher conferences (family ↔ school), when siblings' friends come over (family ↔ peers), when church friends are also school friends."
      },
      {
        term: "Exosystem",
        definition: "Settings that INDIRECTLY affect the child — the child is NOT directly in these settings, but they influence the child's microsystems.",
        example: "Parent's workplace policies (parental leave affects time with child), school board decisions, community resources, parent's social network."
      },
      {
        term: "Macrosystem",
        definition: "The BROADEST level — cultural values, laws, customs, economic conditions, and societal attitudes that influence all other systems.",
        example: "Cultural attitudes about gender roles, national education policies, poverty rates, laws about child welfare, societal views on discipline."
      },
      {
        term: "Chronosystem",
        definition: "CHANGES OVER TIME — historical events, life transitions, and the cumulative effect of experiences across the lifespan.",
        example: "Growing up during COVID, parents' divorce, moving to a new city, the child's own developmental changes over time."
      },
      {
        term: "Evolutionary/Ethological Theories",
        definition: "Apply natural selection and adaptation to human behavior. Focus on FUNCTIONS of behaviors — how did this behavior ensure survival of the species? Examine patterns across human societies.",
        example: "Kindchenschema (baby schema) — infant features (big eyes, round face) trigger caregiving in adults. Attachment behaviors evolved to keep infants close to protectors."
      },
      {
        term: "Comparing All Four Theory Groups",
        definition: "Psychoanalytic: Early relationships shape development (vague, hard to test). Learning: Observable behavior & reinforcement (neglects cognition). Social Cognitive: Child's active interpretation (limited biology). Ecological: Multiple levels of influence (broad, hard to test).",
        example: "Same behavior, different explanations: A child shares toys. Psychoanalytic: Strong superego. Learning: Reinforced for sharing. Social Cognitive: Interpreted the situation as requiring sharing. Ecological: Culture values generosity."
      }
    ],
    quiz: [
      {
        q: "Jamie's toy is taken by another child whose intentions are unclear. Jamie says 'She took it on purpose to be mean!' Jamie's friend says 'Maybe she didn't see you playing with it.' Jamie is demonstrating:",
        options: ["Selman's egocentric role-taking", "Dodge's hostile attributional bias", "Erikson's initiative vs. guilt", "Bronfenbrenner's exosystem"],
        correct: 1,
        explanation: "Jamie assumes hostile intent in an AMBIGUOUS situation — this is Dodge's hostile attributional bias. Jamie's friend, who considers an innocent explanation, shows more advanced social information processing."
      },
      {
        q: "A parent loses their job (which the child doesn't directly experience), leading to family stress that affects the child's behavior at school. This chain of influence operates through Bronfenbrenner's:",
        options: ["Microsystem", "Mesosystem", "Exosystem", "Macrosystem"],
        correct: 2,
        explanation: "The exosystem includes settings the child is NOT directly in (parent's workplace) that nevertheless INDIRECTLY affect them through their microsystems (family stress → school behavior)."
      },
      {
        q: "Which theory emphasizes the importance of CONTEXT in child development? (This was on Exam 1!)",
        options: ["Evolutionary", "Learning", "Ecological", "Cognitive development"],
        correct: 2,
        explanation: "Ecological theories (especially Bronfenbrenner) emphasize multiple levels of contextual influence on development. This was actually asked on Exam 1 — the correct answer was Ecological, not Cognitive development."
      },
      {
        q: "A researcher presents children with stories about ambiguous peer interactions and codes their responses for complexity of reasoning about multiple perspectives. This study best exemplifies which theoretical approach?",
        options: ["Psychoanalytic", "Learning", "Social cognitive", "Ecological"],
        correct: 2,
        explanation: "Studying how children REASON about social situations and take multiple perspectives is the hallmark of social cognitive theories (like Selman's role-taking research)."
      },
      {
        q: "A researcher conducts clinical interviews about adults' early childhood relationships and how they affect current romantic anxiety. This research best exemplifies which theory?",
        options: ["Psychoanalytic", "Learning", "Social cognitive", "Ecological"],
        correct: 0,
        explanation: "Exploring how early relational conflicts shape later patterns of intimacy and anxiety through clinical interviews is classic psychoanalytic research."
      }
    ]
  },
  {
    id: "attachment",
    title: "Attachment",
    session: "12",
    icon: "🤝",
    color: "#f59e0b",
    concepts: [
      {
        term: "Attachment",
        definition: "An EMOTIONAL BOND with a specific person that ENDURES across space and time. Usually discussed regarding infant-caregiver relationships, but also occurs in adulthood.",
        example: "Not just any positive interaction — it's a lasting, specific bond. A baby can be friendly with many people but is ATTACHED to specific caregivers."
      },
      {
        term: "Harlow's Surrogate Mother Study",
        definition: "Challenged the behaviorist view that attachment = food association. Infant rhesus monkeys preferred the CLOTH surrogate mother over the WIRE mother that provided food. Conclusion: Attachment is based on COMFORT and SECURITY, not nourishment.",
        example: "When frightened, baby monkeys ran to the cloth mother for comfort — even if the wire mother was their food source. Contact comfort > food."
      },
      {
        term: "Bowlby's Attachment Theory",
        definition: "Children are BIOLOGICALLY PREDISPOSED to develop attachments to increase survival. Key concepts: SECURE BASE (caregiver provides security for exploration) and SAFE HAVEN (caregiver provides comfort when threatened). Nature AND nurture — innate basis but quality depends on caregiving experiences.",
        example: "A toddler at a playground ventures away to explore but keeps looking back at mom (secure base). When a dog approaches, the toddler runs back to mom (safe haven)."
      },
      {
        term: "Internal Working Model",
        definition: "A mental representation of the SELF, ATTACHMENT FIGURES, and RELATIONSHIPS in general, constructed from experiences with caregivers. Guides expectations for ALL future relationships throughout life.",
        example: "Child with responsive caregiver develops IWM: 'I am worthy of love, others can be trusted.' Child with unresponsive caregiver: 'I'm not worth caring for, others will let me down.'"
      },
      {
        term: "Strange Situation (Ainsworth)",
        definition: "A laboratory procedure with separations from and reunions with the caregiver, plus interactions with a stranger. The KEY observation is how the infant reacts at REUNION — not separation. This classifies attachment quality.",
        example: "Steps include: infant explores with parent → stranger enters → parent leaves → parent returns. The REUNION behavior reveals the attachment classification."
      },
      {
        term: "Secure Attachment (~50%)",
        definition: "Uses caregiver as SECURE BASE for exploration. May be distressed at separation. At reunion: GLAD to see caregiver, comforted quickly, returns to play. Predicted by SENSITIVE, RESPONSIVE caregiving.",
        example: "Baby plays with toys, occasionally checking back on mom. Cries when mom leaves. When mom returns, reaches up to be held, calms quickly, and goes back to playing."
      },
      {
        term: "Insecure-Resistant (~10%)",
        definition: "CLINGY from the start — doesn't explore. VERY distressed at separation. At reunion: approaches caregiver but then RESISTS comfort (pushes away while clinging). Predicted by INCONSISTENT, INTRUSIVE caregiving.",
        example: "Baby stays right next to mom, doesn't explore toys. Cries intensely when mom leaves. When mom returns, reaches up to be held but then arches away and squirms — wants comfort but rejects it."
      },
      {
        term: "Insecure-Avoidant (~15%)",
        definition: "IGNORES caregiver from the start. Not particularly distressed at separation. At reunion: AVOIDS or ignores caregiver. Equally comforted by stranger and caregiver. Predicted by EMOTIONALLY DISTANT, UNRESPONSIVE caregiving.",
        example: "Baby plays independently, doesn't check on mom. Barely reacts when mom leaves. When mom returns, doesn't look up or approach — acts like she's not important."
      },
      {
        term: "Disorganized/Disoriented (~15%)",
        definition: "NO consistent strategy for coping. Confused, CONTRADICTORY behavior — approach AND withdraw simultaneously. May freeze or appear dazed. Predicted by FRIGHTENING, frightened, or absent caregiving. Higher rates in maltreated infants.",
        example: "Baby walks toward mom while looking away with a fearful expression, then suddenly freezes mid-step, then falls to the floor."
      },
      {
        term: "Developmental Progression of Attachment",
        definition: "Indiscriminate attachments (6 weeks–7 months: social reciprocity, Still Face) → Specific attachment (7-9 months: stranger anxiety, separation anxiety) → Multiple attachments (9-18 months).",
        example: "At 4 months, a baby smiles at everyone. At 8 months, the same baby cries when a stranger holds them (stranger anxiety) and when mom leaves (separation anxiety)."
      },
      {
        term: "Attachment Outcomes & Earned Security",
        definition: "Secure attachment predicts: higher social competence, fewer externalizing problems (aggression), fewer internalizing problems (anxiety/depression). BUT foundations are NOT fate — 'earned security' means insecure children CAN become secure adults through therapy and positive relationships.",
        example: "Intergenerational transmission: Securely attached adults are more sensitive parents → their children are more likely to be securely attached. But the cycle CAN be broken."
      },
      {
        term: "Childcare & Attachment",
        definition: "Childcare typically does NOT interfere with attachment. 15-month-olds in childcare are just as likely to be securely attached. High quality childcare can COMPENSATE for less sensitive caregiving. Low quality (high child:caregiver ratio) can be detrimental.",
        example: "Working parents don't need to worry — quality of care matters more than whether the child is in childcare at all."
      }
    ],
    quiz: [
      {
        q: "In Harlow's study, infant monkeys spent most of their time with the cloth surrogate mother even when the WIRE mother provided food. This finding challenged which theoretical view of attachment?",
        options: ["Bowlby's evolutionary theory", "Ainsworth's Strange Situation classification", "The behaviorist view that attachment is based on food/conditioning", "Erikson's trust vs. mistrust stage"],
        correct: 2,
        explanation: "Behaviorists argued the mother-infant bond was classically conditioned through food (mother = conditioned stimulus paired with food). Harlow showed comfort/security mattered more than food."
      },
      {
        q: "In the Strange Situation, a baby clings to her mother and doesn't explore. When her mother leaves, she cries intensely. When her mother returns, she reaches out but then pushes away and can't be comforted. This infant is MOST likely classified as:",
        options: ["Secure", "Insecure-Avoidant", "Insecure-Resistant", "Disorganized"],
        correct: 2,
        explanation: "Insecure-Resistant: clingy (doesn't explore), very distressed at separation, and at reunion seeks contact BUT resists comfort — the hallmark ambivalent/resistant pattern."
      },
      {
        q: "According to Bowlby, an 'internal working model' of attachment is:",
        options: ["The physical growth pattern of the infant brain", "A mental representation of self and relationships that guides future relationship expectations", "The Strange Situation procedure used to measure attachment", "A conscious decision about how to interact with caregivers"],
        correct: 1,
        explanation: "The internal working model is a mental representation of self, attachment figures, and relationships generally. It's built from early caregiving experiences and guides expectations throughout life."
      },
      {
        q: "A Romanian orphanage study compared children adopted before vs. after 6 months of age. Researchers concluded that nature and nurture work together to influence developmental outcomes. This illustrates:",
        options: ["Sensitive periods are irrelevant", "Nurture is more important than nature", "Both timing of experience (nature's sensitive periods) and quality of caregiving (nurture) matter", "Epigenetic changes are not related to environment"],
        correct: 2,
        explanation: "The Romanian orphanage study (also on Exam 1!) showed that earlier adoption led to better outcomes — demonstrating that nature (sensitive periods in brain development) and nurture (quality caregiving) work together."
      }
    ]
  },
  {
    id: "parenting",
    title: "Parents & Family",
    session: "13",
    icon: "👨‍👩‍👧",
    color: "#ef4444",
    concepts: [
      {
        term: "Family Structure vs. Family Dynamics",
        definition: "STRUCTURE = number of and relationships among people in a household (who lives there). DYNAMICS = the way family members INTERACT through relationships (how they relate). Changes in structure affect dynamics and child outcomes.",
        example: "Structure: single parent, two parents, grandparent-led. Dynamics: warm vs. cold parenting, high vs. low conflict between parents."
      },
      {
        term: "Changes in U.S. Family Structure",
        definition: "Five major changes: (1) More single/unmarried parents (70% married), (2) Older first-time parents (27.5 yrs avg), (3) More children with grandparents (~10%), (4) Smaller families, (5) More fluid structures (50% of marriages end in divorce). Repeated transitions → more child behavior problems.",
        example: "Older parents generally = more planned births, more education, more financial resources, more positive parenting. But grandparent caregivers often face financial constraints."
      },
      {
        term: "Warmth-Coldness Dimension",
        definition: "WARM parents: Affectionate, supportive, enjoy children, discipline focuses on changing BEHAVIOR not rejecting CHILD, less harsh physical punishment. COLD parents: Few feelings of affection, don't enjoy children, complain about them. Warm = better outcomes.",
        example: "Warm: 'I love you, but hitting your sister is not OK. Let's talk about other ways to handle frustration.' Cold: 'You're such a bad kid. Go to your room.'"
      },
      {
        term: "Demandingness-Permissiveness Dimension",
        definition: "DEMANDING parents: Impose rules, watch children closely, expect appropriate/mature behavior. PERMISSIVE parents: Few or no rules, less supervision, intervene only when essential.",
        example: "Demanding: Clear bedtime, homework expectations, monitoring screen time. Permissive: 'Whatever you want' — no set rules about much of anything."
      },
      {
        term: "Authoritative Parenting (High Warmth + High Demand)",
        definition: "BEST outcomes. Relationship built on mutual trust and respect. Both perspectives honored. Communication flows both ways. Results: self-reliance, high self-esteem, high social competence, high achievement motivation.",
        example: "'I understand you want to stay up late, but you need sleep for school. Let's compromise — you can read in bed for 15 extra minutes.'"
      },
      {
        term: "Authoritarian Parenting (Low Warmth + High Demand)",
        definition: "Relationship is about CONTROL. Differing perspectives not allowed. Communication flows ONE way. Results: less social/academic competence, more depression and aggression. Sons → hostile/defiant. Daughters → low independence.",
        example: "'Because I said so. No arguing. You will do exactly what I say or there will be consequences.'"
      },
      {
        term: "Permissive Parenting (High Warmth + Low Demand)",
        definition: "Relationship INDULGES the child. Entitlement. Little control. Results: impulsive, low self-control, deviant behaviors, low achievement. BUT high in self-confidence and social competence.",
        example: "'Oh sweetie, of course you can have ice cream for dinner. I just want you to be happy!'"
      },
      {
        term: "Uninvolved/Neglectful Parenting (Low Warmth + Low Demand)",
        definition: "WORST outcomes. No real relationship. No communication. No parenting. Results: disturbed attachment, later peer problems, internalizing AND externalizing problems, substance use, risky behavior.",
        example: "Parent is physically present but emotionally absent — doesn't know child's friends, school performance, or daily activities."
      },
      {
        term: "Cultural Considerations in Parenting",
        definition: "Parenting style outcomes vary by CULTURE and ETHNICITY. Spain/Brazil: permissive → better outcomes. African American families (esp. low-income): restrictive parenting may → better outcomes. Authoritative is not universally 'best.'",
        example: "In some collectivist cultures, what Western researchers might code as 'authoritarian' may reflect appropriate cultural expectations and not produce negative outcomes."
      },
      {
        term: "Bidirectionality",
        definition: "Parent-child influence goes BOTH ways. The child's characteristics (temperament, behavior) influence parenting, not just vice versa. A difficult child may elicit harsher parenting; an easy child may elicit warmer parenting.",
        example: "A child with a difficult temperament may frustrate parents → more negative parenting → worse child behavior → cycle continues. The child is not just a passive recipient."
      }
    ],
    quiz: [
      {
        q: "A parent sets firm, clear rules and enforces them consistently, explains the reasoning behind rules, and encourages the child's input in family decisions. This parent's style is:",
        options: ["Authoritarian", "Authoritative", "Permissive", "Uninvolved"],
        correct: 1,
        explanation: "Authoritative = high warmth (encourages input, explains reasoning) + high demand (firm rules, consistent enforcement). The key distinction from authoritarian is that communication flows BOTH ways."
      },
      {
        q: "Research shows that in some African American families (especially low-income), more restrictive parenting is associated with better child outcomes than in European American families. This finding highlights:",
        options: ["Authoritative parenting is always best", "Cultural context affects the impact of parenting styles", "Only permissive parenting works cross-culturally", "Parenting has no real effect on child outcomes"],
        correct: 1,
        explanation: "This is a key nuance: parenting style outcomes are moderated by cultural and socioeconomic context. What's 'best' depends on the specific context the family lives in."
      },
      {
        q: "Which parenting style is associated with the WORST overall child outcomes?",
        options: ["Authoritarian", "Permissive", "Authoritative", "Uninvolved/Neglectful"],
        correct: 3,
        explanation: "Uninvolved/Neglectful (low warmth + low demand) produces the worst outcomes: disturbed attachment, peer problems, internalizing + externalizing problems, substance use, and risky behavior."
      }
    ]
  },
  {
    id: "peers",
    title: "Peers & Friendship",
    session: "14",
    icon: "👫",
    color: "#8b5cf6",
    concepts: [
      {
        term: "Friendship (Definition)",
        definition: "A relationship that is INTIMATE, RECIPROCATED, and POSITIVE. All three elements are required — one-sided liking isn't friendship.",
        example: "Two children who enjoy each other's company, seek each other out, and help each other = friends. A child who follows another around but isn't liked back = not friendship."
      },
      {
        term: "Children's Choice of Friends",
        definition: "'Birds of a feather flock together' — children befriend peers who are SIMILAR (age, gender, interests, personality). Also: friendly/prosocial behavior, proximity, racial/ethnic similarity (but can be cross-racial).",
        example: "Not 'opposites attract.' Kids pick friends who like the same things, behave similarly, and are nearby."
      },
      {
        term: "Friendship & Perspective-Taking (Selman)",
        definition: "Age-related changes in friendships are tied to qualitative shifts in PERSPECTIVE-TAKING abilities. Young children are egocentric → friendships focus on their own needs. With maturity → friendships involve considering BOTH parties' needs, mutual understanding, trust, and intimacy.",
        example: "Young child: 'A best friend is someone who plays with me.' Older child: 'A best friend is someone who understands me and I understand them — we support each other.'"
      },
      {
        term: "Functions of Friendships",
        definition: "Support and validation; buffer against unpleasant experiences; help develop social skills. Gender differences: girls desire more closeness/dependency, more upset by betrayal, more co-rumination. Gender-atypical youth have more difficulty forming friendships. Few gender differences in stability.",
        example: "Having even one good friend can protect a child from the negative effects of bullying and peer rejection."
      },
      {
        term: "Peer Socialization vs. Peer Selection",
        definition: "Two competing hypotheses for why friends are similar. SOCIALIZATION: Peers influence you → you become more like them. SELECTION: You choose friends who are already like you. Both likely operate — hard to disentangle.",
        example: "Does a teen start drinking because their friends drink (socialization)? Or do they choose friends who drink because they already wanted to (selection)? Probably both."
      },
      {
        term: "Sociometric Status",
        definition: "A research method for measuring peer acceptance using two dimensions: SOCIAL PREFERENCE (liked vs. disliked) and SOCIAL IMPACT (noticed vs. unnoticed). Classifies children into 5 categories.",
        example: "Researchers ask every child in a class 'Who do you like most?' and 'Who do you like least?' then use the pattern of nominations to classify each child."
      },
      {
        term: "Popular (Sociometric)",
        definition: "Highly LIKED and highly IMPACTFUL. Tend to be prosocial and have leadership skills. Note: 'popular' here means genuinely liked, not necessarily 'cool' in the social hierarchy sense.",
        example: "Child who everyone wants to sit with at lunch, who helps others, and who other kids look to for direction during group activities."
      },
      {
        term: "Rejected (Sociometric)",
        definition: "Low acceptance, HIGH rejection, high impact. Often aggressive and disruptive. These children are very NOTICED but actively DISLIKED.",
        example: "Child who other kids specifically name as someone they don't want on their team — noticed and known, but negatively."
      },
      {
        term: "Neglected (Sociometric)",
        definition: "LOW social impact — few positive OR negative ratings. Not especially liked or disliked; they simply go UNNOTICED. This is the key difference from rejected.",
        example: "Child who nobody mentions when asked about likes or dislikes — they fly under the radar entirely."
      },
      {
        term: "Controversial (Sociometric)",
        definition: "Very HIGH impact but AVERAGE preference. Liked by quite a few AND disliked by quite a few — polarizing.",
        example: "Class clown who some kids think is hilarious and fun, while other kids find them annoying and disruptive."
      },
      {
        term: "Cross-Cultural Sociometric Findings",
        definition: "In MOST countries: rejected children = aggressive/disruptive; popular children = prosocial/leadership skills. EXCEPTION: Shy Chinese children (especially in RURAL areas) are NOT rejected — culture encourages reserved behavior. Western cultures encourage independence and self-assertion.",
        example: "Being shy leads to rejection in Western classrooms but acceptance in traditional Chinese rural classrooms."
      }
    ],
    quiz: [
      {
        q: "A child receives very few nominations — neither positive nor negative — from classmates. This child's sociometric classification is MOST likely:",
        options: ["Popular", "Rejected", "Neglected", "Controversial"],
        correct: 2,
        explanation: "Neglected = low social impact — few positive OR negative ratings. They go unnoticed, unlike rejected children who are actively disliked (high impact, low preference)."
      },
      {
        q: "A group of friends all start using a particular slang. One explanation is that the friends influenced each other to adopt the language (peer ___). Another is that kids who already used similar language were drawn together (peer ___).",
        options: ["selection; socialization", "socialization; selection", "assimilation; accommodation", "impact; preference"],
        correct: 1,
        explanation: "Peer socialization = peers influence your behavior. Peer selection = you choose peers who match your existing behavior. Both operate, making causality hard to disentangle."
      },
      {
        q: "Research in China found that shy, reserved children are more accepted by peers than in Western countries. This finding illustrates:",
        options: ["Sociometric methods are invalid in non-Western contexts", "Cultural values shape which behaviors lead to peer acceptance or rejection", "Chinese children are naturally more shy", "Shyness is always positive for development"],
        correct: 1,
        explanation: "Cultural context matters: Western cultures value assertiveness, so shy children may be rejected. Traditional Chinese culture values reserve and compliance, so shy children may be accepted."
      }
    ]
  },
  {
    id: "gender",
    title: "Gender Development",
    session: "15",
    icon: "⚧",
    color: "#06b6d4",
    concepts: [
      {
        term: "Gender Identity",
        definition: "One's self-identification as a particular gender. Established around 2.5–3 years when children can self-label ('I'm a boy/girl'). This is when they KNOW which gender category they belong to.",
        example: "A 3-year-old confidently says 'I'm a girl!' — they have established a gender identity."
      },
      {
        term: "Gender Constancy",
        definition: "Understanding that gender remains the SAME regardless of superficial changes in appearance, clothing, or activities. Develops around AGE 6. Before this, children may think changing clothes changes your gender.",
        example: "Before constancy: 'If a boy wears a dress, he becomes a girl.' After constancy: 'He's still a boy even if he wears a dress — gender doesn't change based on what you wear.'"
      },
      {
        term: "Gender Stereotypes: Developmental Sequence",
        definition: "Children learn gender stereotypes in a predictable ORDER: Appearances first → Toys and activities → Personal/social attributes → Roles. This progression goes from most concrete/visible to most abstract.",
        example: "First they learn 'girls wear dresses' (appearance), then 'girls play with dolls' (toys), then 'girls are gentle' (traits), then 'women are nurses' (roles)."
      },
      {
        term: "Gender Segregation",
        definition: "Strong preference for same-gender playmates, especially prominent in PRESCHOOL years and continuing into middle childhood. One of the most robust findings in gender development.",
        example: "In a preschool classroom, boys cluster with boys and girls cluster with girls for free play — even when adults don't enforce any separation."
      },
      {
        term: "Gender-Role Flexibility vs. Intensification",
        definition: "In ADOLESCENCE, two opposing processes: FLEXIBILITY = recognizing gender roles as social conventions that can be changed. INTENSIFICATION = heightened concerns about adhering to traditional gender roles. Both can occur simultaneously.",
        example: "A teen might intellectually understand that boys can dance (flexibility) but still feel social pressure to play football instead (intensification)."
      },
      {
        term: "Gender Schema Theory",
        definition: "Children form MENTAL FRAMEWORKS (schemas) about gender that guide what they pay ATTENTION to and how they behave. They actively seek gender-relevant information and filter experiences through their gender schema.",
        example: "'This person is a girl. I'm a girl. So I should pay attention to what she's doing and do the same things.' Children preferentially attend to and remember same-gender models."
      },
      {
        term: "Social Cognitive Theory (Gender)",
        definition: "Children learn gender roles through OBSERVATION, REINFORCEMENT, and MODELING. Self-efficacy beliefs about gender-typed activities influence what children pursue. Based on Bandura's framework.",
        example: "'My mom taught me to cook, and I'm proud because I'm good at it.' The child learned through modeling (observation) and developed self-efficacy through practice."
      },
      {
        term: "Social Identity Theory (Gender)",
        definition: "Children identify with their gender IN-GROUP and FAVOR it. Being a member of a gender group shapes behavior and attitudes. 'We are the same, so let's be friends.'",
        example: "A girl preferring to play with other girls simply because 'we're all girls and girls stick together' — in-group bias based on gender category."
      },
      {
        term: "Developmental Intergroup Theory (DIT)",
        definition: "When a social category (like gender) is made SALIENT by the ENVIRONMENT — through labeling, visual markers, and segregation — children develop stereotypes and prejudices about those groups. Environmental salience drives categorization, not just innate preference.",
        example: "When schools say 'Good morning boys and girls,' use gender-segregated lines, and have 'boys vs. girls' activities, they make gender highly salient → more stereotyping."
      },
      {
        term: "Intersex/DSD Conditions",
        definition: "Differences/Disorders of Sex Development — conditions where biological sex characteristics don't fit typical male/female categories. Main types: chromosomal (e.g., XXY) and hormonal (e.g., AIS = androgen insensitivity syndrome, CAH = congenital adrenal hyperplasia).",
        example: "CAH: genetic females exposed to higher androgens prenatally — may show more masculine-typed play preferences. AIS: genetic males insensitive to androgens — may develop female-typical appearance."
      }
    ],
    quiz: [
      {
        q: "A 4-year-old girl watches a female teacher demonstrate a puzzle but ignores a male teacher demonstrating the same puzzle. This child's behavior is BEST explained by:",
        options: ["Social identity theory", "Gender schema theory", "Developmental intergroup theory", "Operant conditioning"],
        correct: 1,
        explanation: "Gender schema theory: the child's gender schema tells her 'this person is a girl like me → I should pay attention to what she does.' She preferentially attends to same-gender models."
      },
      {
        q: "A school organizes activities as 'boys vs. girls,' uses gender-labeled lines, and hangs posters showing 'what boys like' and 'what girls like.' According to Developmental Intergroup Theory, this environment will:",
        options: ["Reduce gender stereotyping by making children aware of gender", "Increase gender stereotyping by making gender an unnecessarily salient category", "Have no effect on gender development", "Only affect boys, not girls"],
        correct: 1,
        explanation: "DIT predicts that making any social category salient through labeling, visual markers, and segregation increases stereotyping and prejudice about those groups."
      },
      {
        q: "A 5-year-old says that if a boy puts on a dress and plays with dolls, the boy 'becomes a girl.' This child has NOT yet developed:",
        options: ["Gender identity", "Gender stereotypes", "Gender constancy", "Gender schema"],
        correct: 2,
        explanation: "Gender constancy (developing around age 6) is understanding that gender remains stable despite superficial changes. This child thinks appearance/behavior changes gender."
      }
    ]
  },
  {
    id: "racial-ethnic",
    title: "Racial/Ethnic Identity",
    session: "16",
    icon: "🌎",
    color: "#d946ef",
    concepts: [
      {
        term: "Ethnic and Racial Identity (ERI)",
        definition: "Beliefs and attitudes about the ethnic or racial groups to which one belongs. Associated with POSITIVE outcomes: higher self-esteem, psychological well-being, fewer emotional/behavioral problems.",
        example: "An adolescent who has explored their cultural heritage, feels pride in their racial identity, and understands what it means to them personally has a strong ERI."
      },
      {
        term: "Marcia's Identity Framework",
        definition: "EXPLORATION + COMMITMENT = Identity Achievement. Four possible statuses: Achievement (explored, committed), Moratorium (exploring, not yet committed), Foreclosure (committed without exploring), Diffusion (neither exploring nor committed).",
        example: "Achievement: 'I've thought deeply about what my heritage means to me and I'm proud of it.' Diffusion: 'I've never really thought about it and it doesn't matter to me.'"
      },
      {
        term: "ERI Development in Infancy",
        definition: "By 3 MONTHS: prefer faces from own racial group. By 9 MONTHS: see people from different races as categorically different. Children notice race VERY early — they are NOT 'colorblind.'",
        example: "This contradicts the popular idea that 'children don't see race.' Research shows they distinguish faces by race as young as 3 months."
      },
      {
        term: "ERI Development in Preschool",
        definition: "Children LABEL themselves by race/ethnicity. View race as CONSTANT (switched-at-birth: race stays with birth parents). Notice skin color differences. Attribute POSITIVE traits to majority groups and NEGATIVE traits to minority groups.",
        example: "A 4-year-old can say 'I'm Black' and understands that race doesn't change if you grow up in a different family. By 3-4, children show same-race preferences and status awareness."
      },
      {
        term: "ERI Development in Adolescence",
        definition: "ERI becomes more CENTRAL to identity. Acculturation and parent-youth acculturation GAPS can cause conflict. Bicultural identity = identifying with both heritage culture and mainstream culture. Higher ERI → better outcomes.",
        example: "A teenager navigating between speaking Spanish at home and English at school, finding pride in both cultures = bicultural identity."
      },
      {
        term: "Cultural Socialization",
        definition: "Messages about cultural values, customs, traditions, history, and PRIDE. The MOST consistently positive type of ethnic-racial socialization. Leads to: stronger ERI, better academic adjustment, adaptive coping, psychological well-being, reduced externalizing behaviors.",
        example: "Parents teaching children about family traditions, celebrating cultural holidays, exposing children to media representing their culture, discussing cultural history."
      },
      {
        term: "Preparation for Bias",
        definition: "Preparing youth for DISCRIMINATION and teaching them how to COPE with it. Outcomes are MIXED — can lead to both externalizing and internalizing behaviors. Context-dependent.",
        example: "Parents discussing: 'Some people may treat you unfairly because of your race. Here's how to handle it if that happens...'"
      },
      {
        term: "Promotion of Mistrust",
        definition: "Communicating the need for WARINESS and DISTRUST of other racial/ethnic groups. Leads to NEGATIVE outcomes: hostility, anxiety.",
        example: "'Don't trust anyone from that group — they're all out to get us.' This creates suspicion and hostility rather than healthy coping."
      },
      {
        term: "Egalitarianism",
        definition: "Emphasizing EQUALITY among racial/ethnic groups. Can include colorblind messages ('race doesn't matter'). Understudied, but some positive effects. However, colorblind approaches can be problematic because children DO see race.",
        example: "'Everyone is equal regardless of race' — well-intentioned but may fail to prepare children for the reality of discrimination."
      }
    ],
    quiz: [
      {
        q: "Research shows that 3-month-old infants prefer faces from their own racial group. This finding is MOST relevant to which debate?",
        options: ["Whether children are naturally 'colorblind'", "Whether gender or race is more salient", "Whether attachment is based on comfort or food", "Whether development is continuous or discontinuous"],
        correct: 0,
        explanation: "This finding directly contradicts the popular belief that children 'don't see race' and are naturally colorblind. They notice racial differences from very early in life."
      },
      {
        q: "Which type of ethnic-racial socialization is associated with the MOST consistently positive outcomes?",
        options: ["Preparation for bias", "Promotion of mistrust", "Cultural socialization", "Egalitarianism"],
        correct: 2,
        explanation: "Cultural socialization — messages about cultural values, traditions, history, and pride — is the most consistently linked to positive outcomes: stronger ERI, academic adjustment, well-being, reduced behavior problems."
      },
      {
        q: "A parent tells their child: 'You can never trust people from other races — they don't have our best interests at heart.' This is an example of:",
        options: ["Cultural socialization", "Preparation for bias", "Promotion of mistrust", "Egalitarianism"],
        correct: 2,
        explanation: "Promotion of mistrust communicates wariness and distrust of other groups. Unlike preparation for bias (which teaches coping with discrimination), this promotes active hostility and suspicion."
      }
    ]
  },
  {
    id: "moral-dev",
    title: "Moral Development",
    session: "17",
    icon: "⚖️",
    color: "#22c55e",
    concepts: [
      {
        term: "Kohlberg's Preconventional Level",
        definition: "Moral reasoning is SELF-CENTERED. Stage 1: Obedience & punishment — what's right = obeying authority to avoid punishment. Stage 2: Instrumental exchange — what's right = what's in MY best interest; 'you scratch my back, I'll scratch yours.'",
        example: "Stage 1: 'Heinz shouldn't steal because he'll go to jail.' Stage 2: 'Heinz should steal because his wife will repay him later.'"
      },
      {
        term: "Kohlberg's Conventional Level",
        definition: "Moral reasoning centers on SOCIAL RELATIONSHIPS and RULES. Stage 3: Mutual expectations — being a 'good person,' meeting others' expectations, maintaining relationships. Stage 4: Social system — upholding laws, fulfilling duties, maintaining social order.",
        example: "Stage 3: 'Heinz should steal because a good husband takes care of his wife.' Stage 4: 'Heinz shouldn't steal because if everyone broke laws, society would collapse.'"
      },
      {
        term: "Kohlberg's Postconventional Level",
        definition: "Moral reasoning centers on IDEALS and PRINCIPLES. Stage 5: Social contract — rules should serve the greatest good; unjust laws should be changed. Stage 6: Universal ethics — self-chosen ethical principles (very rare — Kohlberg eventually stopped scoring it).",
        example: "Stage 5: 'Heinz should steal because human life is more valuable than property rights, and a just society would recognize this.'"
      },
      {
        term: "Social Domain Theory",
        definition: "Developed by Smetana, Killen, Turiel, Nucci (1980s). Three distinct DOMAINS of social reasoning: Moral, Social Conventional, and Personal. Children differentiate these domains from a YOUNG age — they're not just blindly following rules.",
        example: "Even preschoolers understand that hitting (moral) is worse than wearing pajamas to school (conventional) and that choosing a snack is personal choice."
      },
      {
        term: "Moral Domain",
        definition: "Concerns WELFARE, JUSTICE, RIGHTS, and EQUALITY. Moral transgressions are seen as WRONG regardless of rules or authority — they're AUTHORITY-INDEPENDENT. They're also seen as generalizable (wrong everywhere).",
        example: "Hitting, stealing, teasing. A child says 'It's wrong to hit even if the teacher says it's OK' — moral rules don't depend on what authority figures say."
      },
      {
        term: "Social Conventional Domain",
        definition: "Concerns GROUP NORMS, social order, and customs. Conventional transgressions are wrong because of RULES or SOCIAL EXPECTATIONS — they ARE authority-dependent. Without the rule, the behavior would be OK.",
        example: "Calling the teacher by first name, wearing pajamas to school, eating with fingers instead of utensils. 'It's OK to wear PJs if the teacher says you can.'"
      },
      {
        term: "Personal/Psychological Domain",
        definition: "Concerns AUTONOMY and PERSONAL CHOICE. These are matters that should be up to the individual, not governed by rules or authority.",
        example: "Choice of snack, choice of playmate, choice of activity during free time. Children believe these should be THEIR decisions."
      },
      {
        term: "Prosocial Development: Outcomes → Intentions → Motivations",
        definition: "Children's moral evaluations become increasingly sophisticated: First they care about OUTCOMES (was the person helped?), then INTENTIONS (did they mean to help/harm?), then MOTIVATIONS (why did they help — genuinely altruistic or self-serving?).",
        example: "14 months: just care about outcome. 3 yrs: prefer accidental harm-doers over intentional ones. 8 yrs: distinguish altruistic vs. egoistic motives — 'they helped just to show off.'"
      },
      {
        term: "Early Prosocial Behavior",
        definition: "By 14 MONTHS, children spontaneously help others reach goals (e.g., handing objects). By ~2 years, they help even without being asked. They want to see the person helped — doesn't matter if THEY were the helper.",
        example: "Warneken & Tomasello (2007): 14-month-olds pick up a dropped object and hand it to an adult struggling to reach it — no reward, no prompting."
      },
      {
        term: "Strategic Prosociality",
        definition: "Around age 5, children begin engaging in prosocial behavior for SELF-BENEFIT through reciprocity. By age 8, children evaluate public giving more negatively ('showing off') and prefer private giving and altruistic over egoistic motivations.",
        example: "5-year-old shares a toy thinking 'if I share now, they'll share with me later.' 8-year-old judges: 'That kid only donated in front of everyone to look good.'"
      }
    ],
    quiz: [
      {
        q: "A child says hitting is always wrong, even if a teacher says it's OK. But the same child says it's fine to wear pajamas to school if the teacher gives permission. This child is distinguishing between:",
        options: ["Preconventional and conventional moral reasoning", "Moral domain and social conventional domain", "Assimilation and accommodation", "Microsystem and macrosystem"],
        correct: 1,
        explanation: "Social Domain Theory: Moral rules (don't hit) are authority-INDEPENDENT — wrong regardless of what authority says. Conventional rules (dress code) are authority-DEPENDENT — OK if authority permits."
      },
      {
        q: "Two children help a classmate pick up dropped books. One did it because they genuinely cared; the other did it because the teacher was watching and they wanted a reward. An 8-year-old observer would likely judge:",
        options: ["Both helpers as equally good", "The genuine helper as better than the reward-seeking helper", "The reward-seeking helper as better because they achieved the same outcome", "Neither helper as good because helping is expected"],
        correct: 1,
        explanation: "By age 8, children can distinguish altruistic vs. egoistic motivations. They evaluate genuine altruism more positively than self-serving prosocial behavior."
      },
      {
        q: "A child at Kohlberg's Stage 3 would likely say Heinz should steal the drug because:",
        options: ["He'll go to jail if he doesn't", "It's in his best interest to keep his wife alive", "A good husband is expected to do whatever it takes to help his wife", "Universal ethical principles require preserving human life"],
        correct: 2,
        explanation: "Stage 3 (conventional level) = mutual expectations and being a 'good person.' The reasoning focuses on what's expected of someone in a given role — 'a good husband would...'"
      }
    ]
  }
];

// ── COMPONENTS ────────────────────────────────────────────────
const modes = ["study", "flashcards", "quiz"];
const modeLabels = { study: "📖 Study", flashcards: "🃏 Flashcards", quiz: "🧪 Quiz" };

function App() {
  const [mode, setMode] = useState("study");
  const [selectedTopic, setSelectedTopic] = useState(null);
  const [quizState, setQuizState] = useState(null);
  const [flashcardState, setFlashcardState] = useState(null);
  const [scores, setScores] = useState({});
  const [masteredCards, setMasteredCards] = useState({});
  const [sidebarOpen, setSidebarOpen] = useState(true);

  useEffect(() => {
    if (selectedTopic && mode === "quiz") startQuiz(selectedTopic);
    if (selectedTopic && mode === "flashcards") startFlashcards(selectedTopic);
  }, [selectedTopic, mode]);

  function startQuiz(topicId) {
    const topic = TOPICS.find(t => t.id === topicId);
    if (!topic) return;
    const shuffled = [...topic.quiz].sort(() => Math.random() - 0.5);
    setQuizState({ questions: shuffled, current: 0, answers: [], showResult: false, selected: null, confirmed: false });
  }

  function startFlashcards(topicId) {
    const topic = TOPICS.find(t => t.id === topicId);
    if (!topic) return;
    const key = topicId;
    const mastered = masteredCards[key] || [];
    const remaining = topic.concepts.filter((_, i) => !mastered.includes(i));
    const cards = remaining.length > 0 ? remaining : topic.concepts;
    const shuffled = cards.sort(() => Math.random() - 0.5);
    setFlashcardState({ cards: shuffled, current: 0, flipped: false, originalIndices: cards.map((c) => topic.concepts.indexOf(c)) });
  }

  function markMastered(topicId, conceptIndex) {
    setMasteredCards(prev => {
      const key = topicId;
      const arr = prev[key] || [];
      if (!arr.includes(conceptIndex)) return { ...prev, [key]: [...arr, conceptIndex] };
      return prev;
    });
  }

  const topic = selectedTopic ? TOPICS.find(t => t.id === selectedTopic) : null;

  return (
    <div style={{ display: "flex", height: "100vh", fontFamily: "'Source Serif 4', Georgia, serif", background: "#0f0f13", color: "#e4e4e7" }}>
      <link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,300;8..60,400;8..60,600;8..60,700&family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet" />
      
      {/* Sidebar */}
      <div style={{
        width: sidebarOpen ? 280 : 60, transition: "width 0.3s ease",
        background: "#18181b", borderRight: "1px solid #27272a",
        display: "flex", flexDirection: "column", overflow: "hidden", flexShrink: 0
      }}>
        <div style={{ padding: sidebarOpen ? "20px 16px 12px" : "20px 8px 12px", borderBottom: "1px solid #27272a", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          {sidebarOpen && <div style={{ fontFamily: "'DM Sans', sans-serif", fontWeight: 700, fontSize: 15, color: "#a1a1aa", letterSpacing: "0.05em", textTransform: "uppercase" }}>PSYC 2700</div>}
          <button onClick={() => setSidebarOpen(!sidebarOpen)} style={{ background: "none", border: "none", color: "#71717a", cursor: "pointer", fontSize: 18, padding: 4 }}>
            {sidebarOpen ? "◀" : "▶"}
          </button>
        </div>
        
        {sidebarOpen && (
          <>
            <div style={{ padding: "12px 16px 8px" }}>
              <div style={{ display: "flex", gap: 4, background: "#27272a", borderRadius: 8, padding: 3 }}>
                {modes.map(m => (
                  <button key={m} onClick={() => setMode(m)}
                    style={{
                      flex: 1, padding: "7px 4px", border: "none", borderRadius: 6, cursor: "pointer",
                      fontFamily: "'DM Sans', sans-serif", fontSize: 12, fontWeight: 600,
                      background: mode === m ? "#3f3f46" : "transparent",
                      color: mode === m ? "#fafafa" : "#71717a",
                      transition: "all 0.2s"
                    }}>
                    {modeLabels[m]}
                  </button>
                ))}
              </div>
            </div>
            
            <div style={{ flex: 1, overflowY: "auto", padding: "8px 12px" }}>
              {TOPICS.map(t => {
                const mastered = (masteredCards[t.id] || []).length;
                const total = t.concepts.length;
                const score = scores[t.id];
                return (
                  <button key={t.id}
                    onClick={() => setSelectedTopic(t.id)}
                    style={{
                      display: "block", width: "100%", textAlign: "left",
                      padding: "10px 12px", marginBottom: 4, border: "none", borderRadius: 8,
                      background: selectedTopic === t.id ? `${t.color}22` : "transparent",
                      cursor: "pointer", transition: "all 0.2s",
                      borderLeft: selectedTopic === t.id ? `3px solid ${t.color}` : "3px solid transparent"
                    }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <span style={{ fontSize: 18 }}>{t.icon}</span>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontFamily: "'DM Sans', sans-serif", fontSize: 13, fontWeight: 600, color: selectedTopic === t.id ? "#fafafa" : "#a1a1aa", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                          {t.title}
                        </div>
                        <div style={{ fontFamily: "'DM Sans', sans-serif", fontSize: 11, color: "#52525b", marginTop: 2 }}>
                          {mode === "flashcards" ? `${mastered}/${total} mastered` : mode === "quiz" && score != null ? `Last: ${score}%` : `${total} concepts`}
                        </div>
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </>
        )}
      </div>

      {/* Main content */}
      <div style={{ flex: 1, overflowY: "auto", padding: "0" }}>
        {!selectedTopic ? (
          <WelcomeScreen setSelectedTopic={setSelectedTopic} setMode={setMode} scores={scores} masteredCards={masteredCards} />
        ) : mode === "study" ? (
          <StudyMode topic={topic} />
        ) : mode === "flashcards" ? (
          <FlashcardMode topic={topic} state={flashcardState} setState={setFlashcardState} markMastered={markMastered} masteredCards={masteredCards} startFlashcards={startFlashcards} />
        ) : (
          <QuizMode topic={topic} state={quizState} setState={setQuizState} scores={scores} setScores={setScores} startQuiz={startQuiz} />
        )}
      </div>
    </div>
  );
}

function WelcomeScreen({ setSelectedTopic, setMode, scores, masteredCards }) {
  const totalConcepts = TOPICS.reduce((s, t) => s + t.concepts.length, 0);
  const totalMastered = Object.values(masteredCards).reduce((s, a) => s + a.length, 0);
  const totalQuizzes = Object.keys(scores).length;
  const avgScore = totalQuizzes > 0 ? Math.round(Object.values(scores).reduce((s, v) => s + v, 0) / totalQuizzes) : null;

  return (
    <div style={{ maxWidth: 800, margin: "0 auto", padding: "60px 32px" }}>
      <div style={{ marginBottom: 48 }}>
        <h1 style={{ fontFamily: "'Source Serif 4', Georgia, serif", fontSize: 42, fontWeight: 700, lineHeight: 1.15, marginBottom: 12, background: "linear-gradient(135deg, #a78bfa, #6366f1, #ec4899)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
          Exam 2 Study Guide
        </h1>
        <p style={{ fontFamily: "'DM Sans', sans-serif", fontSize: 16, color: "#71717a", lineHeight: 1.6 }}>
          Sessions 09–17 · Social Cognition through Moral Development
        </p>
      </div>

      {/* Stats */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12, marginBottom: 40 }}>
        {[
          { label: "Concepts", value: totalConcepts, sub: "across 9 topics" },
          { label: "Flashcards Mastered", value: `${totalMastered}/${totalConcepts}`, sub: `${Math.round(totalMastered/totalConcepts*100)}% complete` },
          { label: "Avg Quiz Score", value: avgScore != null ? `${avgScore}%` : "—", sub: totalQuizzes > 0 ? `${totalQuizzes} quizzes taken` : "none yet" }
        ].map((s, i) => (
          <div key={i} style={{ background: "#18181b", border: "1px solid #27272a", borderRadius: 12, padding: "20px 16px" }}>
            <div style={{ fontFamily: "'DM Sans', sans-serif", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.08em", color: "#52525b", marginBottom: 8 }}>{s.label}</div>
            <div style={{ fontFamily: "'Source Serif 4', serif", fontSize: 28, fontWeight: 700, color: "#fafafa" }}>{s.value}</div>
            <div style={{ fontFamily: "'DM Sans', sans-serif", fontSize: 12, color: "#52525b", marginTop: 4 }}>{s.sub}</div>
          </div>
        ))}
      </div>

      <h2 style={{ fontFamily: "'DM Sans', sans-serif", fontSize: 14, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.06em", color: "#52525b", marginBottom: 16 }}>Choose a topic to begin</h2>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10 }}>
        {TOPICS.map(t => (
          <button key={t.id} onClick={() => { setSelectedTopic(t.id); setMode("study"); }}
            style={{
              background: "#18181b", border: "1px solid #27272a", borderRadius: 12, padding: "20px 16px",
              cursor: "pointer", textAlign: "left", transition: "all 0.2s",
              borderTop: `3px solid ${t.color}`
            }}
            onMouseEnter={e => { e.currentTarget.style.borderColor = t.color; e.currentTarget.style.transform = "translateY(-2px)"; }}
            onMouseLeave={e => { e.currentTarget.style.borderColor = "#27272a"; e.currentTarget.style.borderTopColor = t.color; e.currentTarget.style.transform = "none"; }}
          >
            <div style={{ fontSize: 24, marginBottom: 8 }}>{t.icon}</div>
            <div style={{ fontFamily: "'DM Sans', sans-serif", fontSize: 14, fontWeight: 600, color: "#e4e4e7" }}>{t.title}</div>
            <div style={{ fontFamily: "'DM Sans', sans-serif", fontSize: 12, color: "#52525b", marginTop: 4 }}>Session {t.session} · {t.concepts.length} concepts</div>
          </button>
        ))}
      </div>
    </div>
  );
}

function StudyMode({ topic }) {
  const [expanded, setExpanded] = useState({});
  const [showAll, setShowAll] = useState(false);

  const toggle = (i) => setExpanded(prev => ({ ...prev, [i]: !prev[i] }));

  return (
    <div style={{ maxWidth: 760, margin: "0 auto", padding: "40px 32px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
        <span style={{ fontSize: 32 }}>{topic.icon}</span>
        <h1 style={{ fontFamily: "'Source Serif 4', serif", fontSize: 32, fontWeight: 700 }}>{topic.title}</h1>
      </div>
      <p style={{ fontFamily: "'DM Sans', sans-serif", fontSize: 13, color: "#71717a", marginBottom: 24 }}>
        Session {topic.session} · {topic.concepts.length} concepts · Click each card to reveal details
      </p>
      <button onClick={() => setShowAll(!showAll)} style={{
        fontFamily: "'DM Sans', sans-serif", fontSize: 13, padding: "8px 16px",
        background: "#27272a", color: "#a1a1aa", border: "none", borderRadius: 8,
        cursor: "pointer", marginBottom: 20
      }}>{showAll ? "Collapse All" : "Expand All"}</button>

      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {topic.concepts.map((c, i) => {
          const open = showAll || expanded[i];
          return (
            <div key={i} onClick={() => toggle(i)} style={{
              background: "#18181b", border: "1px solid #27272a", borderRadius: 12,
              cursor: "pointer", transition: "all 0.2s", overflow: "hidden",
              borderLeft: `3px solid ${topic.color}`
            }}>
              <div style={{ padding: "16px 20px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                <div style={{ fontFamily: "'DM Sans', sans-serif", fontSize: 15, fontWeight: 600, color: "#fafafa" }}>{c.term}</div>
                <span style={{ color: "#52525b", fontSize: 14, transition: "transform 0.2s", transform: open ? "rotate(180deg)" : "rotate(0)" }}>▼</span>
              </div>
              {open && (
                <div style={{ padding: "0 20px 20px", borderTop: "1px solid #27272a22" }}>
                  <p style={{ fontFamily: "'Source Serif 4', serif", fontSize: 15, lineHeight: 1.7, color: "#d4d4d8", marginTop: 12 }}>{c.definition}</p>
                  {c.example && (
                    <div style={{ marginTop: 12, padding: "12px 16px", background: `${topic.color}11`, borderRadius: 8, borderLeft: `3px solid ${topic.color}44` }}>
                      <div style={{ fontFamily: "'DM Sans', sans-serif", fontSize: 11, fontWeight: 600, textTransform: "uppercase", color: topic.color, marginBottom: 6 }}>Example</div>
                      <p style={{ fontFamily: "'Source Serif 4', serif", fontSize: 14, lineHeight: 1.6, color: "#a1a1aa" }}>{c.example}</p>
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function FlashcardMode({ topic, state, setState, markMastered, masteredCards, startFlashcards }) {
  if (!state || !state.cards.length) return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", flexDirection: "column", gap: 16 }}>
      <div style={{ fontSize: 48 }}>🎉</div>
      <h2 style={{ fontFamily: "'DM Sans', sans-serif", fontSize: 20, fontWeight: 600 }}>All cards mastered!</h2>
      <button onClick={() => startFlashcards(topic.id)} style={{
        fontFamily: "'DM Sans', sans-serif", padding: "10px 24px", background: topic.color,
        color: "white", border: "none", borderRadius: 8, cursor: "pointer", fontSize: 14, fontWeight: 600
      }}>Review All Again</button>
    </div>
  );

  const card = state.cards[state.current];
  const idx = topic.concepts.indexOf(card);
  const flip = () => setState(p => ({ ...p, flipped: !p.flipped }));
  const next = () => setState(p => ({ ...p, current: Math.min(p.current + 1, p.cards.length - 1), flipped: false }));
  const prev = () => setState(p => ({ ...p, current: Math.max(p.current - 1, 0), flipped: false }));
  const master = () => { markMastered(topic.id, idx); next(); };

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", padding: 32 }}>
      <div style={{ fontFamily: "'DM Sans', sans-serif", fontSize: 13, color: "#52525b", marginBottom: 24 }}>
        {state.current + 1} / {state.cards.length} remaining · {(masteredCards[topic.id] || []).length} mastered
      </div>
      
      <div onClick={flip} style={{
        width: "100%", maxWidth: 560, minHeight: 320, background: "#18181b",
        border: `1px solid ${topic.color}44`, borderRadius: 16, cursor: "pointer",
        display: "flex", flexDirection: "column", justifyContent: "center", padding: "40px 36px",
        transition: "all 0.3s", position: "relative", overflow: "hidden",
        boxShadow: `0 0 40px ${topic.color}11`
      }}>
        <div style={{ position: "absolute", top: 16, right: 20, fontFamily: "'DM Sans', sans-serif", fontSize: 11, color: "#52525b", textTransform: "uppercase", letterSpacing: "0.08em" }}>
          {state.flipped ? "Definition" : "Term"} · click to flip
        </div>
        {!state.flipped ? (
          <h2 style={{ fontFamily: "'Source Serif 4', serif", fontSize: 28, fontWeight: 700, textAlign: "center", color: "#fafafa" }}>
            {card.term}
          </h2>
        ) : (
          <div>
            <p style={{ fontFamily: "'Source Serif 4', serif", fontSize: 16, lineHeight: 1.7, color: "#d4d4d8", marginBottom: 16 }}>{card.definition}</p>
            {card.example && (
              <p style={{ fontFamily: "'Source Serif 4', serif", fontSize: 14, lineHeight: 1.6, color: "#71717a", fontStyle: "italic", borderTop: "1px solid #27272a", paddingTop: 12 }}>
                {card.example}
              </p>
            )}
          </div>
        )}
      </div>

      <div style={{ display: "flex", gap: 12, marginTop: 24, flexWrap: "wrap", justifyContent: "center" }}>
        <button onClick={prev} disabled={state.current === 0} style={{
          fontFamily: "'DM Sans', sans-serif", padding: "10px 20px", background: "#27272a",
          color: state.current === 0 ? "#3f3f46" : "#a1a1aa", border: "none", borderRadius: 8,
          cursor: state.current === 0 ? "not-allowed" : "pointer", fontSize: 13, fontWeight: 600
        }}>← Back</button>
        <button onClick={master} style={{
          fontFamily: "'DM Sans', sans-serif", padding: "10px 20px", background: "#166534",
          color: "#86efac", border: "none", borderRadius: 8, cursor: "pointer", fontSize: 13, fontWeight: 600
        }}>✓ Got it</button>
        <button onClick={() => setState(p => ({ ...p, flipped: false, current: p.current }))} style={{
          fontFamily: "'DM Sans', sans-serif", padding: "10px 20px", background: "#7f1d1d",
          color: "#fca5a5", border: "none", borderRadius: 8, cursor: "pointer", fontSize: 13, fontWeight: 600
        }}>✗ Still learning</button>
        <button onClick={next} disabled={state.current >= state.cards.length - 1} style={{
          fontFamily: "'DM Sans', sans-serif", padding: "10px 20px", background: "#27272a",
          color: state.current >= state.cards.length - 1 ? "#3f3f46" : "#a1a1aa", border: "none", borderRadius: 8,
          cursor: state.current >= state.cards.length - 1 ? "not-allowed" : "pointer", fontSize: 13, fontWeight: 600
        }}>Next →</button>
      </div>
    </div>
  );
}

function QuizMode({ topic, state, setState, scores, setScores, startQuiz }) {
  if (!state) return null;

  const q = state.questions[state.current];

  if (state.showResult) {
    const correct = state.answers.filter(a => a.correct).length;
    const total = state.questions.length;
    const pct = Math.round(correct / total * 100);
    return (
      <div style={{ maxWidth: 640, margin: "0 auto", padding: "60px 32px" }}>
        <h2 style={{ fontFamily: "'Source Serif 4', serif", fontSize: 32, fontWeight: 700, textAlign: "center", marginBottom: 8 }}>
          {pct >= 80 ? "🎉" : pct >= 60 ? "👍" : "📚"} {correct} / {total}
        </h2>
        <p style={{ fontFamily: "'DM Sans', sans-serif", fontSize: 16, color: "#71717a", textAlign: "center", marginBottom: 32 }}>
          {pct >= 80 ? "Great job!" : pct >= 60 ? "Getting there — review the ones you missed." : "Keep studying — you'll get there!"}
        </p>
        
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {state.answers.map((a, i) => (
            <div key={i} style={{
              background: "#18181b", border: `1px solid ${a.correct ? "#16a34a44" : "#dc262644"}`, borderRadius: 12, padding: "16px 20px"
            }}>
              <div style={{ display: "flex", alignItems: "flex-start", gap: 10 }}>
                <span style={{ fontSize: 16, marginTop: 2 }}>{a.correct ? "✅" : "❌"}</span>
                <div>
                  <p style={{ fontFamily: "'DM Sans', sans-serif", fontSize: 14, fontWeight: 500, color: "#d4d4d8", marginBottom: 8 }}>{a.question}</p>
                  {!a.correct && (
                    <p style={{ fontFamily: "'DM Sans', sans-serif", fontSize: 13, color: "#71717a" }}>
                      <span style={{ color: "#dc2626" }}>Your answer: {a.yourAnswer}</span><br/>
                      <span style={{ color: "#16a34a" }}>Correct: {a.correctAnswer}</span>
                    </p>
                  )}
                  <p style={{ fontFamily: "'Source Serif 4', serif", fontSize: 13, color: "#52525b", marginTop: 6, fontStyle: "italic" }}>{a.explanation}</p>
                </div>
              </div>
            </div>
          ))}
        </div>

        <button onClick={() => startQuiz(topic.id)} style={{
          display: "block", margin: "32px auto 0", fontFamily: "'DM Sans', sans-serif", padding: "12px 32px",
          background: topic.color, color: "white", border: "none", borderRadius: 8, cursor: "pointer", fontSize: 14, fontWeight: 600
        }}>Retake Quiz</button>
      </div>
    );
  }

  function selectAnswer(idx) {
    if (state.confirmed) return;
    setState(p => ({ ...p, selected: idx }));
  }

  function confirmAnswer() {
    const isCorrect = state.selected === q.correct;
    const answer = {
      question: q.q,
      correct: isCorrect,
      yourAnswer: q.options[state.selected],
      correctAnswer: q.options[q.correct],
      explanation: q.explanation
    };
    setState(p => ({ ...p, confirmed: true, answers: [...p.answers, answer] }));
  }

  function nextQuestion() {
    if (state.current >= state.questions.length - 1) {
      const correct = state.answers.filter(a => a.correct).length;
      const pct = Math.round(correct / state.questions.length * 100);
      setScores(p => ({ ...p, [topic.id]: pct }));
      setState(p => ({ ...p, showResult: true }));
    } else {
      setState(p => ({ ...p, current: p.current + 1, selected: null, confirmed: false }));
    }
  }

  return (
    <div style={{ maxWidth: 680, margin: "0 auto", padding: "60px 32px" }}>
      <div style={{ fontFamily: "'DM Sans', sans-serif", fontSize: 13, color: "#52525b", marginBottom: 24 }}>
        Question {state.current + 1} of {state.questions.length}
      </div>
      
      {/* Progress bar */}
      <div style={{ height: 4, background: "#27272a", borderRadius: 2, marginBottom: 32 }}>
        <div style={{ height: "100%", width: `${((state.current) / state.questions.length) * 100}%`, background: topic.color, borderRadius: 2, transition: "width 0.3s" }} />
      </div>

      <h3 style={{ fontFamily: "'Source Serif 4', serif", fontSize: 20, fontWeight: 600, lineHeight: 1.5, marginBottom: 24, color: "#fafafa" }}>{q.q}</h3>

      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {q.options.map((opt, i) => {
          let bg = "#18181b";
          let borderColor = "#27272a";
          let textColor = "#d4d4d8";
          if (state.confirmed) {
            if (i === q.correct) { bg = "#052e1644"; borderColor = "#16a34a"; textColor = "#86efac"; }
            else if (i === state.selected && i !== q.correct) { bg = "#450a0a44"; borderColor = "#dc2626"; textColor = "#fca5a5"; }
          } else if (i === state.selected) {
            bg = `${topic.color}22`; borderColor = topic.color; textColor = "#fafafa";
          }
          return (
            <button key={i} onClick={() => selectAnswer(i)} style={{
              display: "block", width: "100%", textAlign: "left",
              padding: "14px 18px", background: bg, border: `1px solid ${borderColor}`,
              borderRadius: 10, cursor: state.confirmed ? "default" : "pointer",
              fontFamily: "'DM Sans', sans-serif", fontSize: 14, color: textColor,
              transition: "all 0.2s", lineHeight: 1.5
            }}>
              <span style={{ fontWeight: 600, marginRight: 10, color: "#52525b" }}>{String.fromCharCode(65 + i)}.</span>
              {opt}
            </button>
          );
        })}
      </div>

      {state.confirmed && (
        <div style={{ marginTop: 20, padding: "16px 20px", background: "#18181b", borderRadius: 12, borderLeft: `3px solid ${state.selected === q.correct ? "#16a34a" : "#dc2626"}` }}>
          <p style={{ fontFamily: "'Source Serif 4', serif", fontSize: 14, lineHeight: 1.6, color: "#a1a1aa" }}>
            {q.explanation}
          </p>
        </div>
      )}

      <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 24, gap: 12 }}>
        {!state.confirmed ? (
          <button onClick={confirmAnswer} disabled={state.selected === null} style={{
            fontFamily: "'DM Sans', sans-serif", padding: "10px 28px",
            background: state.selected === null ? "#27272a" : topic.color,
            color: state.selected === null ? "#52525b" : "white",
            border: "none", borderRadius: 8, cursor: state.selected === null ? "not-allowed" : "pointer",
            fontSize: 14, fontWeight: 600
          }}>Check Answer</button>
        ) : (
          <button onClick={nextQuestion} style={{
            fontFamily: "'DM Sans', sans-serif", padding: "10px 28px",
            background: topic.color, color: "white", border: "none", borderRadius: 8,
            cursor: "pointer", fontSize: 14, fontWeight: 600
          }}>{state.current >= state.questions.length - 1 ? "See Results" : "Next Question →"}</button>
        )}
      </div>
    </div>
  );
}

export default App;