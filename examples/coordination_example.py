#!/usr/bin/env python3
"""
Multi-Agent Coordination Example for ECON Framework

This example demonstrates the coordination mechanisms
and Nash Equilibrium concepts in ECON.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def demonstrate_bne_coordination():
    """
    Demonstrate Bayesian Nash Equilibrium coordination concepts.
    """
    
    print("🤝 Multi-Agent Coordination with BNE")
    print("=" * 50)
    
    print("📚 Theoretical Background:")
    print("1. Belief Formation: Each agent develops beliefs about the problem")
    print("2. Strategic Interaction: Agents consider others' likely actions")
    print("3. Equilibrium: No agent can improve by unilaterally changing strategy")
    print("4. Convergence: Iterative updates lead to stable coordination")
    
    simulate_belief_evolution()
    
    print("\n🔄 Two-Stage Process:")
    print("Stage 1 - Individual Analysis:")
    print("  - Each agent processes the question independently")
    print("  - Forms initial beliefs and solution approach")
    print("  - Generates candidate response")
    
    print("\nStage 2 - BNE Coordination:")
    print("  - Agents share belief representations (not explicit messages)")
    print("  - Update beliefs based on equilibrium computation")
    print("  - Converge to coordinated solution")
    
    demonstrate_coordination_benefits()

def simulate_belief_evolution():
    """
    Simulate how agent beliefs evolve during coordination.
    """
    
    print("\n📊 Belief Evolution Simulation")
    print("-" * 30)
    
    n_agents = 3
    n_rounds = 5
    
    np.random.seed(42)
    beliefs = np.random.uniform(0, 1, (n_agents, n_rounds))
    
    for round_idx in range(1, n_rounds):
        for agent_idx in range(n_agents):
            other_beliefs = np.mean([beliefs[i, round_idx-1] 
                                   for i in range(n_agents) if i != agent_idx])
            beliefs[agent_idx, round_idx] = (
                0.7 * beliefs[agent_idx, round_idx-1] + 
                0.3 * other_beliefs +
                np.random.normal(0, 0.05)  # Small noise
            )
    
    print("Agent belief evolution across rounds:")
    print("Round", end="")
    for i in range(n_rounds):
        print(f"\t{i+1}", end="")
    print()
    
    for agent_idx in range(n_agents):
        print(f"Agent {agent_idx+1}", end="")
        for round_idx in range(n_rounds):
            print(f"\t{beliefs[agent_idx, round_idx]:.3f}", end="")
        print()
    
    final_std = np.std(beliefs[:, -1])
    print(f"\nFinal belief variance: {final_std:.4f}")
    if final_std < 0.1:
        print("✅ Agents achieved consensus!")
    else:
        print("⚠️ More rounds needed for full convergence")

def demonstrate_coordination_benefits():
    """
    Show the benefits of coordination vs independent reasoning.
    """
    
    print("\n🎯 Coordination vs Independent Reasoning")
    print("-" * 45)
    
    scenarios = [
        ("Independent Agents", 0.65, "Each agent works alone"),
        ("Message Passing", 0.72, "Agents share explicit messages"),
        ("ECON Coordination", 0.83, "BNE-based belief coordination")
    ]
    
    print("Approach\t\tAccuracy\tDescription")
    print("-" * 60)
    for approach, accuracy, description in scenarios:
        print(f"{approach:<20}\t{accuracy:.2f}\t\t{description}")
    
    print("\n📈 Key Advantages of ECON:")
    print("- Higher accuracy through coordinated reasoning")
    print("- Lower communication cost (beliefs vs messages)")
    print("- Theoretical convergence guarantees")
    print("- Scalable to many agents")

def show_reward_breakdown():
    """
    Demonstrate the three-component reward system.
    """
    
    print("\n💰 Reward System Components")
    print("=" * 35)
    
    example_scenario = {
        "correct_answer": True,
        "agent_consistency": 0.85,  # High agreement between agents
        "response_quality": 0.75,   # Good reasoning quality
        "coordination_quality": 0.80  # Effective collaboration
    }
    
    ts_reward = 1.0 if example_scenario["correct_answer"] else 0.0
    al_reward = example_scenario["agent_consistency"]
    cc_reward = (example_scenario["response_quality"] + 
                 example_scenario["coordination_quality"]) / 2
    
    weights = {"TS": 0.5, "AL": 0.3, "CC": 0.2}
    total_reward = (weights["TS"] * ts_reward + 
                   weights["AL"] * al_reward + 
                   weights["CC"] * cc_reward)
    
    print("Component\tWeight\tValue\tContribution")
    print("-" * 45)
    print(f"TS (Task Specific)\t{weights['TS']:.1f}\t{ts_reward:.2f}\t{weights['TS']*ts_reward:.3f}")
    print(f"AL (Action Likelihood)\t{weights['AL']:.1f}\t{al_reward:.2f}\t{weights['AL']*al_reward:.3f}")
    print(f"CC (Collaborative)\t{weights['CC']:.1f}\t{cc_reward:.2f}\t{weights['CC']*cc_reward:.3f}")
    print("-" * 45)
    print(f"Total Reward\t\t\t\t{total_reward:.3f}")
    
    print("\n📝 Reward Explanations:")
    print("- TS: Correctness of final answer (0 or 1)")
    print("- AL: Consistency between agent responses (0-1)")
    print("- CC: Quality of collaboration and reasoning (0-1)")

def create_coordination_diagram():
    """
    Create a simple diagram showing coordination flow.
    """
    
    print("\n🎨 Creating Coordination Flow Diagram")
    print("-" * 40)
    
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.set_title("Stage 1: Individual Belief Formation")
        ax1.text(0.5, 0.8, "Question", ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        agents_x = [0.2, 0.5, 0.8]
        for i, x in enumerate(agents_x):
            ax1.text(x, 0.5, f"Agent {i+1}", ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen"))
            ax1.arrow(0.5, 0.75, x-0.5, -0.2, head_width=0.02, 
                     head_length=0.03, fc='black', ec='black')
        
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        
        ax2.set_title("Stage 2: BNE Coordination")
        
        for i, x1 in enumerate(agents_x):
            ax2.text(x1, 0.7, f"Agent {i+1}", ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="lightcoral"))
            for j, x2 in enumerate(agents_x):
                if i != j:
                    ax2.plot([x1, x2], [0.7, 0.7], 'k--', alpha=0.3)
        
        ax2.text(0.5, 0.3, "Coordinated\nSolution", ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="gold"))
        
        for x in agents_x:
            ax2.arrow(x, 0.65, 0.5-x, -0.3, head_width=0.02, 
                     head_length=0.03, fc='red', ec='red')
        
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        plt.tight_layout()
        
        output_dir = Path("examples/outputs")
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "coordination_flow.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ Coordination diagram saved to: examples/outputs/coordination_flow.png")
        
    except ImportError:
        print("⚠️ Matplotlib not available - skipping diagram creation")
    except Exception as e:
        print(f"❌ Error creating diagram: {e}")

if __name__ == "__main__":
    demonstrate_bne_coordination()
    show_reward_breakdown()
    create_coordination_diagram() 