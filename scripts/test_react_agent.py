#!/usr/bin/env python3
"""
Test Script: ReAct Clinical Agent

Demonstrates the agentic framework with:
- Tool-based reasoning
- Transparent thought process
- Step-by-step execution

Run: python scripts/test_react_agent.py
"""

import sys
sys.path.insert(0, '.')

from src.agents import (
    ClinicalAgent,
    PatientInfo,
    create_clinical_agent,
)


def print_section(title: str):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")


def test_single_assessment():
    """Test single patient assessment with reasoning trace."""
    print_section("Single Assessment with Reasoning Trace")
    
    # Create agent (use mock=True for testing without GCP)
    agent = create_clinical_agent(mock=False)
    
    # Create patient
    patient = PatientInfo(
        age=55,
        sex="male",
        symptoms=["persistent cough", "weight loss", "fatigue"],
        symptom_duration="3 weeks",
        risk_factors=["smoker", "30 pack years"],
        presenting_complaint="Persistent cough with unintentional weight loss"
    )
    
    print("Patient Information:")
    print(patient.to_task_description())
    
    print("\n" + "-"*40)
    print("Running ReAct Agent...")
    print("-"*40 + "\n")
    
    # Run assessment
    result = agent.assess(patient)
    
    # Display reasoning trace
    if result.reasoning_trace:
        print("\n📋 REASONING TRACE:")
        print("-"*40)
        for step in result.reasoning_trace.steps:
            print(f"\n🤔 Thought {step.step_number}:")
            print(f"   {step.thought[:200]}...")
            if step.action:
                print(f"\n⚡ Action: {step.action}")
                if step.action_input:
                    print(f"   Input: {step.action_input}")
            if step.observation:
                print(f"\n👁️ Observation:")
                print(f"   {step.observation[:200]}...")
    
    # Display results
    print("\n" + "="*40)
    print("📊 ASSESSMENT RESULT")
    print("="*40)
    
    print(f"\n🎯 Risk Level: {result.risk_level.value.upper()}")
    print(f"⏰ Urgency: {result.get_urgency_display()}")
    print(f"\n📝 Summary: {result.summary}")
    
    if result.recommended_actions:
        print("\n✅ Recommended Actions:")
        for action in result.recommended_actions:
            print(f"   • {action}")
    
    if result.red_flags:
        print("\n⚠️ Red Flags:")
        for flag in result.red_flags:
            print(f"   • {flag}")
    
    if result.citations:
        print(f"\n📚 Citations: {', '.join(result.citations)}")
    
    print(f"\n📈 Confidence: {result.confidence*100:.0f}%")
    print(f"🔧 Tools Used: {', '.join(result.tools_used)}")
    print(f"📊 Steps Taken: {result.steps_taken}")


def test_streaming_assessment():
    """Test streaming assessment output."""
    print_section("Streaming Assessment")
    
    agent = create_clinical_agent(mock=False)
    
    patient = PatientInfo(
        age=62,
        sex="female",
        symptoms=["breast lump", "skin changes"],
        presenting_complaint="New breast lump noticed 2 weeks ago"
    )
    
    print("Patient:", patient.presenting_complaint)
    print("\nStreaming reasoning steps:\n")
    
    for step in agent.assess_streaming(patient):
        step_type = step.get("type")
        
        if step_type == "thinking":
            print(f"🤔 Step {step['step']}: Thinking...")
        elif step_type == "thought":
            print(f"💭 Thought: {step['thought'][:100]}...")
        elif step_type == "action":
            print(f"⚡ Action: {step['action']}")
        elif step_type == "observation":
            print(f"👁️ Observation received")
        elif step_type == "final_answer":
            print(f"\n✅ Final Answer Generated")
        elif step_type == "assessment":
            print(f"\n📊 Risk: {step['assessment']['risk_level']}")


def test_available_tools():
    """Display available tools."""
    print_section("Available Tools")
    
    agent = create_clinical_agent(mock=False)
    tools = agent.get_available_tools()
    
    for tool in tools:
        print(f"🔧 {tool['name']}")
        print(f"   {tool['description'][:100]}...")
        params = tool['parameters']['properties']
        print(f"   Parameters: {', '.join(params.keys())}")
        print()


def test_quick_assessment():
    """Test quick symptom-based assessment."""
    print_section("Quick Assessment (Symptoms Only)")
    
    agent = create_clinical_agent(mock=False)
    
    # Quick assessment from symptoms
    result = agent.assess_quick(
        symptoms=["haemoptysis", "persistent cough"],
        age=58
    )
    
    print(f"Symptoms: haemoptysis, persistent cough")
    print(f"Age: 58")
    print(f"\nResult:")
    print(f"  Risk: {result.risk_level.value}")
    print(f"  Urgency: {result.urgency.value}")
    print(f"  Summary: {result.summary}")


if __name__ == "__main__":
    print("\n" + "🏥"*20)
    print(" NG12 Clinical Agent - ReAct Framework Demo")
    print("🏥"*20)
    
    # Show available tools
    test_available_tools()
    
    # Run single assessment
    test_single_assessment()
    
    # Quick assessment
    test_quick_assessment()
    
    # Streaming (optional)
    # test_streaming_assessment()
    
    print("\n" + "="*60)
    print(" Demo Complete!")
    print("="*60 + "\n")
