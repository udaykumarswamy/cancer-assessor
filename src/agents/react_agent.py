"""
ReAct Agent Implementation

Implements the ReAct (Reasoning + Acting) pattern for clinical assessment.
The agent iteratively reasons about the problem, takes actions via tools,
and observes results until reaching a conclusion.

Interview Discussion Points:
---------------------------
1. Why ReAct pattern?
   - Transparent reasoning: Every step is documented
   - Auditable: Critical for clinical applications
   - Interruptible: Can pause and resume
   - Debuggable: Easy to see where things went wrong

2. Agent loop:
   Thought → Action → Observation → Thought → ... → Final Answer

3. Clinical advantages:
   - Shows reasoning chain for medical decisions
   - Cites specific guidelines
   - Can be reviewed by clinicians
   - Provides evidence trail

4. Comparison with other patterns:
   - ReAct vs Chain-of-Thought: ReAct adds tool use
   - ReAct vs Plan-and-Execute: More flexible, handles uncertainty
   - ReAct vs Function Calling: More structured reasoning
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import re

from src.agents.tools import ClinicalTools, Tool, ToolResult
from src.config.logging_config import get_logger

logger = get_logger("react_agent")


class AgentState(Enum):
    """States of the ReAct agent."""
    READY = "ready"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    FINISHED = "finished"
    ERROR = "error"


@dataclass
class ThoughtStep:
    """A single thought in the reasoning chain."""
    step_number: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_prompt_format(self) -> str:
        """Format for inclusion in prompt."""
        lines = [f"Thought {self.step_number}: {self.thought}"]
        if self.action:
            lines.append(f"Action {self.step_number}: {self.action}")
            if self.action_input:
                lines.append(f"Action Input {self.step_number}: {json.dumps(self.action_input)}")
        if self.observation:
            lines.append(f"Observation {self.step_number}: {self.observation}")
        return "\n".join(lines)


@dataclass
class AgentTrace:
    """Complete trace of agent reasoning and actions."""
    task: str
    steps: List[ThoughtStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    state: AgentState = AgentState.READY
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    finished_at: Optional[str] = None
    error: Optional[str] = None
    
    def add_step(self, step: ThoughtStep):
        """Add a step to the trace."""
        self.steps.append(step)
    
    def get_reasoning_chain(self) -> str:
        """Get formatted reasoning chain."""
        return "\n\n".join(step.to_prompt_format() for step in self.steps)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task": self.task,
            "steps": [
                {
                    "step": s.step_number,
                    "thought": s.thought,
                    "action": s.action,
                    "action_input": s.action_input,
                    "observation": s.observation[:200] + "..." if s.observation and len(s.observation) > 200 else s.observation
                }
                for s in self.steps
            ],
            "final_answer": self.final_answer,
            "state": self.state.value,
            "total_steps": len(self.steps),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error
        }


class ReActAgent:
    """
    ReAct Agent for Clinical Assessment.
    
    Implements the Reasoning + Acting pattern:
    1. THINK: Reason about the current state and what to do
    2. ACT: Choose a tool and execute it
    3. OBSERVE: Process the tool result
    4. REPEAT until task is complete
    
    Usage:
        agent = ReActAgent(tools, llm)
        
        # Run assessment
        result = agent.run(
            "Assess a 55 year old male with persistent cough and weight loss"
        )
        
        # Access reasoning trace
        print(result.trace.get_reasoning_chain())
    """
    
    SYSTEM_PROMPT = """You are a clinical decision support agent helping healthcare professionals 
assess patients for potential cancer referral according to NICE NG12 guidelines.

You have access to tools to search guidelines, check symptoms, and calculate risk.
You must use these tools to gather evidence before making recommendations.

IMPORTANT RULES:
1. Always search guidelines before making clinical recommendations
2. Check for red flags when symptoms are concerning  
3. Calculate risk level based on NG12 criteria
4. Cite specific guideline sections in your final answer
5. Be clear about urgency levels (2-week pathway vs routine)
6. This is decision SUPPORT - final decisions are made by clinicians

You operate in a loop of: Thought → Action → Observation

Format your responses EXACTLY as:

Thought: [Your reasoning about what to do next]
Action: [tool_name]
Action Input: {"param1": "value1", "param2": "value2"}

OR when you have enough information:

Thought: [Your final reasoning]
Final Answer: [Your complete clinical assessment with recommendations and citations]"""

    REACT_PROMPT = """Based on the task and previous steps, continue your reasoning.

Task: {task}

Available Tools:
{tools_description}

Previous Steps:
{previous_steps}

Continue with the next thought. Remember to use tools to gather evidence.
If you have enough information, provide your Final Answer.

Your response:"""

    def __init__(
        self,
        tools: ClinicalTools,
        llm,
        max_steps: int = 10,
        verbose: bool = True
    ):
        """
        Initialize the ReAct agent.
        
        Args:
            tools: ClinicalTools instance
            llm: LLM instance for reasoning
            max_steps: Maximum reasoning steps
            verbose: Print steps to console
        """
        self.tools = tools
        self.llm = llm
        self.max_steps = max_steps
        self.verbose = verbose
    
    def run(self, task: str) -> 'AgentResult':
        """
        Run the agent on a task.
        
        Args:
            task: The clinical assessment task
            
        Returns:
            AgentResult with final answer and trace
        """
        logger.info(f"Starting ReAct agent for task: {task[:100]}...")
        logger.debug(f"Max steps configured: {self.max_steps}")
        logger.debug(f"Verbose mode: {self.verbose}")
        
        trace = AgentTrace(task=task, state=AgentState.THINKING)
        logger.debug("AgentTrace initialized")
        
        try:
            # Build tools description
            logger.debug("Formatting tools description...")
            tools_desc = self._format_tools_description()
            logger.debug(f"Tools available for agent")
            
            step_num = 0
            while step_num < self.max_steps:
                step_num += 1
                logger.info(f"\n=== Step {step_num} ===")
                
                # Build prompt with history
                logger.debug("Building prompt with reasoning history...")
                previous_steps = trace.get_reasoning_chain() if trace.steps else "None yet."
                
                prompt = self.REACT_PROMPT.format(
                    task=task,
                    tools_description=tools_desc,
                    previous_steps=previous_steps
                )
                logger.debug("Prompt constructed")
                
                # Get LLM response
                logger.debug("Sending request to LLM for reasoning...")
                trace.state = AgentState.THINKING
                response = self.llm.generate(
                    prompt=prompt,
                    system_instruction=self.SYSTEM_PROMPT,
                    temperature=0.1,
                    max_tokens=2048
                )
                logger.debug("LLM response received")
                
                raw_output = response.content
                if self.verbose:
                    logger.debug(f"Step {step_num} raw output: {raw_output[:200]}...")
                
                # Parse the response
                logger.debug("Parsing LLM response...")
                thought, action, action_input, final_answer = self._parse_response(raw_output)
                logger.debug(f"Response parsed - Action: {action}, Has final answer: {final_answer is not None}")
                
                # Create step
                logger.debug(f"Creating ThoughtStep - Thought: {thought[:100]}...")
                step = ThoughtStep(
                    step_number=step_num,
                    thought=thought
                )
                
                # Check for final answer
                if final_answer:
                    logger.info(f"Final answer generated at step {step_num}")
                    trace.add_step(step)
                    trace.final_answer = final_answer
                    trace.state = AgentState.FINISHED
                    trace.finished_at = datetime.now().isoformat()
                    logger.debug(f"Final answer length: {len(final_answer)} chars")
                    
                    if self.verbose:
                        logger.info(f"Agent finished after {step_num} steps")
                    
                    return AgentResult(
                        success=True,
                        answer=final_answer,
                        trace=trace
                    )
                
                # Execute action
                if action:
                    logger.info(f"Step {step_num}: Executing action '{action}'")
                    step.action = action
                    step.action_input = action_input
                    logger.debug(f"Action input: {action_input}")
                    
                    trace.state = AgentState.ACTING
                    if self.verbose:
                        logger.info(f"Step {step_num}: Executing {action}")
                    
                    # Execute tool
                    logger.debug(f"Calling tool: {action} with params: {action_input}")
                    tool_result = self.tools.execute_tool(action, **(action_input or {}))
                    logger.debug(f"Tool execution completed")
                    
                    trace.state = AgentState.OBSERVING
                    observation = tool_result.to_string()
                    step.observation = observation
                    logger.debug(f"Observation recorded (length: {len(observation)} chars)")
                    
                    if self.verbose:
                        logger.debug(f"Observation: {observation[:200]}...")
                else:
                    logger.debug("No action detected in this step")
                
                trace.add_step(step)
                logger.debug(f"Step {step_num} complete and added to trace")
            
            # Max steps reached
            logger.warning(f"Max steps ({self.max_steps}) reached without final answer")
            logger.warning(f"Total steps completed: {len(trace.steps)}")
            trace.state = AgentState.ERROR
            trace.error = "Maximum reasoning steps reached"
            
            return AgentResult(
                success=False,
                answer="Unable to complete assessment within step limit",
                trace=trace,
                error="Max steps reached"
            )
            
        except Exception as e:
            logger.error(f"Agent execution error at step {step_num}: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            trace.state = AgentState.ERROR
            trace.error = str(e)
            
            return AgentResult(
                success=False,
                answer=None,
                trace=trace,
                error=str(e)
            )
    
    def _format_tools_description(self) -> str:
        """Format tools for inclusion in prompt."""
        logger.debug("Formatting tools description for prompt...")
        lines = []
        all_tools = self.tools.get_all_tools()
        logger.debug(f"Total tools available: {len(all_tools)}")
        
        for i, tool in enumerate(all_tools, 1):
            logger.debug(f"Formatting tool {i}: {tool.name}")
            params = ", ".join(
                f"{p.name}: {p.type}" + (" (optional)" if not p.required else "")
                for p in tool.parameters
            )
            lines.append(f"- {tool.name}({params})")
            lines.append(f"  Description: {tool.description[:200]}...")
        
        result = "\n".join(lines)
        logger.debug(f"Tools formatting complete, description length: {len(result)} chars")
        return result
    
    def _parse_response(
        self,
        response: str
    ) -> Tuple[str, Optional[str], Optional[Dict], Optional[str]]:
        """
        Parse agent response into components.
        
        Returns:
            (thought, action, action_input, final_answer)
        """
        logger.debug("Parsing LLM response...")
        thought = ""
        action = None
        action_input = None
        final_answer = None
        
        # Extract thought
        logger.debug("Attempting to extract thought...")
        thought_match = re.search(r'Thought(?:\s*\d*)?:\s*(.+?)(?=Action|Final Answer|$)', response, re.DOTALL | re.IGNORECASE)
        if thought_match:
            thought = thought_match.group(1).strip()
            logger.debug(f"Thought extracted (length: {len(thought)} chars)")
        else:
            logger.debug("No thought found in response")
        
        # Check for final answer
        logger.debug("Checking for final answer...")
        final_match = re.search(r'Final Answer:\s*(.+)$', response, re.DOTALL | re.IGNORECASE)
        if final_match:
            final_answer = final_match.group(1).strip()
            logger.info(f"Final answer detected (length: {len(final_answer)} chars)")
            return thought, None, None, final_answer
        
        # Extract action
        logger.debug("Extracting action...")
        action_match = re.search(r'Action(?:\s*\d*)?:\s*(\w+)', response, re.IGNORECASE)
        if action_match:
            action = action_match.group(1).strip()
            logger.debug(f"Action extracted: {action}")
        else:
            logger.debug("No action found in response")
        
        # Extract action input
        logger.debug("Extracting action input...")
        input_match = re.search(r'Action Input(?:\s*\d*)?:\s*(\{.+?\})', response, re.DOTALL | re.IGNORECASE)
        if input_match:
            try:
                action_input = json.loads(input_match.group(1))
                logger.debug(f"Action input parsed successfully: {list(action_input.keys())}")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse action input JSON: {e}")
                # Try to extract individual parameters
                action_input = {}
        else:
            logger.debug("No action input found in response")
        
        logger.debug(f"Parse result - Thought: {len(thought)} chars, Action: {action}, Input: {action_input is not None}")
        return thought, action, action_input, final_answer


@dataclass
class AgentResult:
    """Result from agent execution."""
    success: bool
    answer: Optional[str]
    trace: AgentTrace
    error: Optional[str] = None
    
    def get_citations(self) -> List[str]:
        """Extract citations from the answer."""
        if not self.answer:
            return []
        
        # Find NG12 citations
        citations = re.findall(r'\[NG12[^\]]+\]', self.answer)
        return citations
    
    def get_tools_used(self) -> List[str]:
        """Get list of tools used."""
        return [step.action for step in self.trace.steps if step.action]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "answer": self.answer,
            "trace": self.trace.to_dict(),
            "citations": self.get_citations(),
            "tools_used": self.get_tools_used(),
            "error": self.error
        }


class StreamingReActAgent(ReActAgent):
    """
    Streaming version of ReAct agent.
    
    Yields steps as they're generated for real-time display.
    """
    
    def run_streaming(self, task: str):
        """
        Run agent with streaming output.
        
        Yields:
            Dict with step information as it's generated
        """
        logger.info(f"Starting streaming ReAct agent for task: {task[:100]}...")
        logger.debug(f"Max steps for streaming: {self.max_steps}")
        
        trace = AgentTrace(task=task, state=AgentState.THINKING)
        logger.debug("AgentTrace initialized for streaming")
        
        logger.debug("Formatting tools for streaming session...")
        tools_desc = self._format_tools_description()
        
        logger.info("Yielding start event")
        yield {"type": "start", "task": task}
        
        step_num = 0
        while step_num < self.max_steps:
            step_num += 1
            logger.info(f"\n=== Streaming Step {step_num} ===")
            
            # Build prompt
            logger.debug("Building prompt with history...")
            previous_steps = trace.get_reasoning_chain() if trace.steps else "None yet."
            prompt = self.REACT_PROMPT.format(
                task=task,
                tools_description=tools_desc,
                previous_steps=previous_steps
            )
            logger.debug("Prompt built")
            
            # Get LLM response
            logger.debug("Yielding thinking event and requesting LLM response...")
            yield {"type": "thinking", "step": step_num}
            
            logger.debug(f"Calling LLM for step {step_num}...")
            response = self.llm.generate(
                prompt=prompt,
                system_instruction=self.SYSTEM_PROMPT,
                temperature=0.1
            )
            logger.debug("LLM response received")
            
            # Parse response
            logger.debug("Parsing LLM response...")
            thought, action, action_input, final_answer = self._parse_response(response.content)
            logger.debug(f"Parsed - Has action: {action is not None}, Has final answer: {final_answer is not None}")
            
            step = ThoughtStep(
                step_number=step_num,
                thought=thought
            )
            logger.debug(f"ThoughtStep created for step {step_num}")
            
            logger.debug(f"Yielding thought event")
            yield {"type": "thought", "step": step_num, "thought": thought}
            
            # Final answer?
            if final_answer:
                logger.info(f"Final answer generated at step {step_num}")
                trace.add_step(step)
                trace.final_answer = final_answer
                trace.state = AgentState.FINISHED
                logger.debug("Trace state set to FINISHED")
                
                logger.debug("Yielding final answer event")
                yield {
                    "type": "final_answer",
                    "step": step_num,
                    "answer": final_answer,
                    "trace": trace.to_dict()
                }
                logger.info(f"Streaming completed successfully in {step_num} steps")
                return
            
            # Execute action
            if action:
                logger.info(f"Step {step_num}: Executing action '{action}'")
                step.action = action
                step.action_input = action_input
                logger.debug(f"Action input: {action_input}")
                
                logger.debug("Yielding action event")
                yield {
                    "type": "action",
                    "step": step_num,
                    "action": action,
                    "input": action_input
                }
                
                logger.debug(f"Executing tool: {action}")
                tool_result = self.tools.execute_tool(action, **(action_input or {}))
                logger.debug("Tool execution completed")
                
                observation = tool_result.to_string()
                step.observation = observation
                logger.debug(f"Observation captured (length: {len(observation)} chars)")
                
                logger.debug("Yielding observation event")
                yield {
                    "type": "observation",
                    "step": step_num,
                    "observation": observation[:500]
                }
            else:
                logger.debug(f"No action found in step {step_num}")
            
            trace.add_step(step)
            logger.debug(f"Step {step_num} added to trace")
        
        # Max steps reached
        logger.warning(f"Maximum streaming steps ({self.max_steps}) reached without final answer")
        logger.warning(f"Total steps completed: {len(trace.steps)}")
        logger.debug("Yielding error event")
        yield {
            "type": "error",
            "message": "Maximum steps reached",
            "trace": trace.to_dict()
        }
