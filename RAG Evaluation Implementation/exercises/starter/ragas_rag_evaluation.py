# RAGAS RAG System Evaluation Framework - Starter Template
# TODO: Complete this script to evaluate RAG systems using the RAGAS framework

# TODO: Import necessary libraries
# Hint: You'll need pandas, numpy, typing, json, time, datetime, matplotlib, seaborn
from webbrowser import get

import pandas as pd
import numpy as np
# TODO: Add remaining imports here
from typing import List, Dict, Tuple, Optional
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# TODO: Import RAGAS components
# Hint: You'll need evaluate function and various metrics from ragas
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas.metrics.collections import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    ContextRelevance,
    Faithfulness,
)

# TODO: Import dataset creation and LLM integration
# Hint: You'll need Dataset from datasets and OpenAI client
from datasets import Dataset
# TODO: Add OpenAI import
import openai
from openai import OpenAI


import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv is optional; we can still read a pre-set env var.
    load_dotenv = None

# Read the key from the environment at runtime (never commit real keys).
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


_client: Optional[OpenAI] = None


def get_client(api_key=OPENAI_API_KEY) -> OpenAI:
    """Create the OpenAI client once and reuse it for every call."""
    global _client
    if _client is None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        _client = OpenAI(
            api_key=api_key,
            base_url="https://openai.vocareum.com/v1" if api_key.startswith("voc") else None
        )
    return _client


METRIC_REGISTRY = {
    "faithfulness": Faithfulness,
    "answer_relevancy": AnswerRelevancy,
    "context_precision": ContextPrecision,
    "context_recall": ContextRecall,
    "context_relevance": ContextRelevance,
    "answer_correctness": AnswerCorrectness
}


# TODO: Define evaluation configurations for different assessment strategies
# Create configurations focusing on different aspects of RAG performance
EVALUATION_CONFIGS = {
    "comprehensive": {
        # TODO: Add all RAGAS metrics for complete evaluation
        "metrics": [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "context_relevance",
            "answer_correctness"
        ],
        "description": "Complete evaluation covering all aspects of RAG performance"
    },
    "retrieval_focused": {
        # TODO: Add context-related metrics only
        "metrics": [
            "context_precision",
            "context_recall",
            "context_relevance"
        ],
        "description": "Evaluation focused on retrieval component quality"
    },
    "generation_focused": {
        # TODO: Add answer-related metrics only
        "metrics": [
            "faithfulness",
            "answer_relevancy",
            "answer_correctness"
        ],
        "description": "Evaluation focused on generation component quality"
    },
    "quick_eval": {
        # TODO: Add essential metrics for fast evaluation
        "metrics": [
            "faithfulness",
            "answer_relevancy",
            "context_precision"
        ],
        "description": "Fast evaluation for iterative development"
    }
}


# TODO: Create test datasets for different domains and complexity levels
# Define realistic evaluation scenarios with questions, ground truth, and contexts
EVALUATION_DATASETS = {
    "technical_qa": {
        "name": "Technical Documentation Q&A",
        "description": "Questions about technical concepts and implementations",
        "test_cases": [
            {
                "question": "What is ChromaDB and what are its main features?",  # TODO: Add technical question about ChromaDB
                "ground_truth": "ChromaDB is an open-source vector database designed for AI applications. Its main features include efficient storage and retrieval of high-dimensional vectors, support for multiple embedding functions, both in-memory and persistent storage options, and optimization for semantic search and RAG implementations.",  # TODO: Add expected correct answer
                "contexts": [
                    "ChromaDB is an open-source vector database designed for AI applications. It provides efficient storage and retrieval of high-dimensional vectors, making it ideal for semantic search, recommendation systems, and RAG implementations.",
                    "ChromaDB supports multiple embedding functions and offers both in-memory and persistent storage options. It's optimized for AI workloads and provides excellent performance for similarity search operations."
                ]  # TODO: Add relevant context documents
            },
            {
                "question": "How does RAG improve upon traditional language model responses?",  # TODO: Add question about RAG systems
                "ground_truth": "RAG (Retrieval-Augmented Generation) improves upon traditional language model responses by combining external knowledge retrieval with generation. This provides more accurate, up-to-date, and contextually relevant responses which reduce hallucinations and improve factual accuracy.",  # TODO: Add expected answer
                "contexts": [
                    "Retrieval-Augmented Generation (RAG) combines the power of large language models with external knowledge retrieval. By retrieving relevant documents before generation, RAG systems can provide more accurate responses.",
                    "RAG systems reduce hallucinations and improve factual accuracy by grounding generated responses in retrieved context. This makes them more reliable for knowledge-intensive tasks."
                ]  # TODO: Add relevant contexts
            },
            {
                "question": "What are vector embeddings and how do they enable semantic search?",  # TODO: Add question about vector embeddings
                "ground_truth": "Vector embeddings are numerical representations of text that capture semantic meaning. They enable semantic search by converting text into high-dimensional vectors where similar concepts are positioned close together in the vector space, allowing for meaning-based rather than keyword-based search.",  # TODO: Add expected answer
                "contexts": [
                    "Vector embeddings are numerical representations of text that capture semantic meaning. Modern embedding models can convert text into high-dimensional vectors where similar concepts are positioned close together in the vector space.",
                    "In vector space, similar concepts are positioned close together, enabling semantic search. This allows for meaning-based retrieval rather than relying solely on keyword matching."
                ]  # TODO: Add relevant contexts
            }
        ]
    },
    "customer_support": {
        "name": "Customer Support FAQ",
        "description": "Common customer service questions and procedures",
        "test_cases": [
            {
                "question": "",  # TODO: Add customer support question
                "ground_truth": "To reset your password, click on the 'Forgot Password' link on the login page, enter your email address, and follow the instructions sent to your email. The reset link expires after 24 hours for your security.",  # TODO: Add expected answer
                "contexts": [
                    "Password reset process: Click 'Forgot Password' on the login page, enter your email address, and follow the email instructions. Reset links expire after 24 hours.",
                    "For security, password reset links are only valid for 24 hours. If expired, you'll need to request a new link."
                ]  # TODO: Add relevant contexts
            },
            {
                "question": "What are your business hours?",  # TODO: Add another support question
                "ground_truth": "Our customer support is available Monday through Friday from 9 AM to 6 PM EST. For urgent technical issues, our emergency support line is available 24/7 for premium customers.",  # TODO: Add expected answer
                "contexts": [
                    "Customer support hours: Monday to Friday, 9 AM to 6 PM EST. Emergency support is available 24/7 for premium customers.",
                    "Premium customers have access to 24/7 emergency support for urgent technical issues outside regular business hours."
                ]  # TODO: Add relevant contexts
            },
            {
                "question": "How can I upgrade my subscription?",  # TODO: Add third support question
                "ground_truth": "You can upgrade your subscription by logging into your account, navigating to the 'Billing' section and selecting 'Upgrade Plan'. Changes take effect immediately, and you'll be prorated for the current billing period.",  # TODO: Add expected answer
                "contexts": [
                    "Subscription upgrade process: Log into your account, go to the 'Billing' section, and select 'Upgrade Plan'. Changes are immediate with prorated billing.",
                    "Billing is prorated when upgrading subscriptions, so you only pay for the remaining time in your current billing period."
                ]  # TODO: Add relevant contexts
            }
        ]
    }
}


class RAGSystemEvaluator:
    """
    Comprehensive RAG system evaluation using RAGAS framework.
    
    TODO: Complete this class to implement a production-ready evaluation system with:
    - RAGAS metric integration and analysis
    - Automated dataset creation and formatting
    - Comprehensive result interpretation and reporting
    - Configuration comparison and optimization
    """
    
    def __init__(self, openai_api_key: str = None):
        """
        Initialize the RAG system evaluator.
        
        TODO: Complete this method to:
        1. Store the OpenAI API key and initialize client
        2. Initialize result storage and dataset management
        3. Print initialization status and available configurations
        
        Args:
            openai_api_key (str): OpenAI API key for LLM-based evaluations
        """
        # TODO: Initialize OpenAI client
        self.openai_client = get_client(openai_api_key) if openai_api_key else get_client()

        # Wire OpenAI into RAGAS via the current factory functions. This allows RAGAS to use the OpenAI LLM and embeddings for evaluation.
        self.llm = llm_factory("gpt-6-luna", client=self.openai_client)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", client=self.openai_client)

        # Instantiate every metric at once, bound to the factories. AnswerRelevancy and AnswerCorrectness need embeddings, 
        # while the others acces it (and maybe ignore it) via **kwargs, so passing it to all is safe.
        self.metrics = {name: cls(llm=self.llm, embeddings=self.embeddings) for name, cls in METRIC_REGISTRY.items()}

        
        # TODO: Initialize storage for evaluation results and datasets
        self.evaluation_results = {}
        self.test_datasets = {}
        
        # TODO: Print initialization status
        print("🔍 RAG System Evaluator initialized")
        print(f"   Available evaluation configs: {list(EVALUATION_CONFIGS.keys())}")
        print(f"   Available test datasets: {list(EVALUATION_DATASETS.keys())}")


    def create_evaluation_dataset(self, dataset_key: str, rag_system_responses: List[Dict]) -> Dataset:
        """
        Create a RAGAS-compatible dataset from test cases and RAG responses.
        
        TODO: Complete this method to:
        1. Validate dataset_key exists in EVALUATION_DATASETS
        2. Extract test cases and format data for RAGAS
        3. Combine questions, contexts, answers, and ground truth
        4. Create and return RAGAS Dataset object
        5. Handle missing responses gracefully
        
        Args:
            dataset_key (str): Key identifying the test dataset
            rag_system_responses (List[Dict]): RAG system responses for each test case
            
        Returns:
            Dataset: RAGAS-compatible dataset for evaluation
        """
        # TODO: Validate dataset exists
        if dataset_key not in EVALUATION_DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_key}")
            
        # TODO: Get dataset configuration and test cases
        dataset_config = EVALUATION_DATASETS[dataset_key]
        test_cases = dataset_config["test_cases"]
        
        print(f"\n📊 Creating evaluation dataset: {dataset_config['name']}")
        print(f"   Description: {dataset_config['description']}")
        print(f"   Number of test cases: {len(test_cases)}")
        
        # TODO: Prepare data in RAGAS format
        # TODO: Process each test case and corresponding RAG response
        # Hint: Loop through test_cases and match with rag_system_responses
        n = min(len(test_cases), len(rag_system_responses))
        if n < len(test_cases):
            print(f"⚠️ Warning: Only {n} RAG system responses provided for {len(test_cases)} test cases. Some cases will be skipped.")

        evaluation_data = {
            "question": [case["question"] for case in test_cases[:n]],
            "contexts": [case["contexts"] for case in test_cases[:n]],
            "answer": [response["answer"] for response in rag_system_responses[:n]],
            "ground_truth": [case["ground_truth"] for case in test_cases[:n]]
        }
            
        # TODO: Create and return RAGAS dataset
        # Hint: Use Dataset.from_dict(evaluation_data)
        dataset = Dataset.from_dict(evaluation_data)
        print(f"✅ Evaluation dataset created with {len(dataset)} entries")
        return dataset
    

    def evaluate_rag_system(self, dataset: Dataset, config_key: str = "comprehensive") -> Dict:
        """
        Evaluate RAG system using specified RAGAS metrics.
        
        TODO: Complete this method to:
        1. Validate configuration exists
        2. Run RAGAS evaluation with specified metrics
        3. Format results with comprehensive analysis
        4. Calculate overall scores and metric breakdowns
        5. Handle evaluation errors gracefully
        
        Args:
            dataset (Dataset): RAGAS-compatible evaluation dataset
            config_key (str): Configuration key for evaluation metrics
            
        Returns:
            Dict: Comprehensive evaluation results with metrics and analysis
        """
        # TODO: Validate configuration
        if config_key not in EVALUATION_CONFIGS:
            raise ValueError(f"Unknown evaluation config: {config_key}")
            
        config = EVALUATION_CONFIGS[config_key]
        
        print(f"\n🧪 Evaluating RAG system with {config_key} configuration")
        print(f"    Description: {config['description']}")
        print(f"    Metrics: {config['metrics']}")

        metric_objects = [self.metrics[name] for name in config["metrics"]]
        
        # TODO: Record start time for performance tracking
        start_time = time.time()
        
        try:
            # TODO: Run RAGAS evaluation
            # Hint: Use evaluate() function with dataset, metrics, and LLM
            results = evaluate(
                dataset=dataset,
                metrics=metric_objects
            )
            
            # TODO: Calculate evaluation time
            evaluation_time = time.time() - start_time
            
            # TODO: Format results for analysis
            formatted_results = {
                "config_used": config_key,
                "evaluation_time": round(evaluation_time, 2),
                "dataset_size": len(dataset),
                "metrics": {},
                "overall_score": 0,
                "detailed_scores": None
            }
            
            # TODO: Extract and format metric scores
            for name in config["metrics"]:
                key = self.metrics[name].name
                values = []
                for row in results.scores:
                    if key in row and row[key] is not None:
                        values.append(getattr(row[key], "value", row[key]))
                if values:
                    formatted_results["metrics"][name] = round(np.mean(values), 4)

            # TODO: Calculate overall score
            scored = list(formatted_results["metrics"].values())
            if scored:
                formatted_results["overall_score"] = round(np.mean(scored), 4)
            
            print(f"✅ Evaluation completed successfully")
            print(f"   Evaluation time: {evaluation_time:.2f} seconds")
            print(f"   Overall score: {formatted_results['overall_score']:.4f}")
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Evaluation failed: {str(e)}")
            return {
                "config_used": config_key,
                "evaluation_time": 0,
                "dataset_size": len(dataset),
                "metrics": {},
                "overall_score": 0,
                "error": str(e)
            }


    def display_evaluation_results(self, results: Dict, dataset_name: str = "Unknown") -> None:
        """
        Display evaluation results in a formatted, comprehensive way.
        
        TODO: Complete this method to:
        1. Display evaluation summary with key metrics
        2. Show detailed metric breakdown with interpretations
        3. Provide performance analysis and insights
        4. Generate actionable recommendations
        5. Format output for readability
        
        Args:
            results (Dict): Evaluation results from evaluate_rag_system
            dataset_name (str): Name of the evaluated dataset
        """
        print(f"\n" + "="*80)
        print(f"📊 RAG SYSTEM EVALUATION RESULTS")
        print(f"="*80)
        
        # TODO: Display evaluation summary
        print(f"\n📋 EVALUATION SUMMARY:")
        # Include: dataset, configuration, size, time, overall score
        print(f"   Dataset: {dataset_name}")
        print(f"   Configuration: {results.get('config_used', 'N/A')}")
        print(f"   Dataset size: {results.get('dataset_size', 0)} examples")
        print(f"   Evaluation time: {results.get('evaluation_time', 0):.2f} seconds")
        print(f"   Overall score: {results.get('overall_score', 0):.4f}")
        
        # TODO: Handle evaluation errors
        if "error" in results:
            print(f"\n❌ EVALUATION ERROR:")
            print(f"   {results['error']}")
            return
        
        # TODO: Display metric breakdown with interpretations
        print(f"\n📈 METRIC BREAKDOWN:")
        # For each metric, show score and interpretation
        for metric_name, score in results.get("metrics", {}).items():
            interpretation = self._interpret_metric_score(metric_name, score)
            print(f"   {metric_name}: {score:.4f} -> {interpretation}")
        
        # TODO: Provide performance analysis
        print(f"\n💡 PERFORMANCE ANALYSIS:")
        # Analyze retrieval vs generation performance
        # Identify strengths and weaknesses
        self._analyze_performance(results)
        
        # TODO: Provide actionable recommendations
        print(f"\n🎯 RECOMMENDATIONS:")
        # Generate specific improvement suggestions
        self._provide_recommendations(results)


    def _interpret_metric_score(self, metric_name: str, score: float) -> str:
        """
        Provide interpretation for individual metric scores.
        
        TODO: Complete this method to:
        1. Define score quality thresholds (excellent, good, fair, poor)
        2. Provide metric-specific interpretations
        3. Return human-readable score explanations
        
        Args:
            metric_name (str): Name of the metric
            score (float): Metric score (typically 0-1)
            
        Returns:
            str: Human-readable interpretation of the score
        """
        # TODO: Define score thresholds
        if score >= 0.8:
            quality = "Excellent"
        # TODO: Add remaining thresholds
        elif score >= 0.6:
            quality = "Good"
        elif score >= 0.4:
            quality = "Fair"
        elif score >= 0.2:
            quality = "Poor"
        else:
            quality = "Very Poor"

        # TODO: Create metric-specific interpretations
        interpretations = {
            "faithfulness": f"{quality} - Generated answers are {'highly' if score >= 0.8 else 'moderately' if score >= 0.6 else 'poorly'} faithful to retrieved context",
            # TODO: Add interpretations for other metrics
            "answer_relevancy": f"{quality} - Generated answers are {'highly' if score >= 0.8 else 'moderately' if score >= 0.6 else 'poorly'} relevant to questions",
            "context_precision": f"{quality} - Retrieved context has {'high' if score >= 0.8 else 'moderate' if score >= 0.6 else 'poor'} precision",
            "context_recall": f"{quality} - Retrieved context has {'high' if score >= 0.8 else 'moderate' if score >= 0.6 else 'poor'} recall",
            "answer_correctness": f"{quality} - Answers are {'highly' if score >= 0.8 else 'moderately' if score >= 0.6 else 'poorly'} correct",
            "context_relevance": f"{quality} - Retrieved context is {'highly' if score >= 0.8 else 'moderately' if score >= 0.6 else 'poorly'} relevant"
        }
        
        return interpretations.get(metric_name, f"{quality} performance")


    def _analyze_performance(self, results: Dict) -> None:
        """
        Provide comprehensive performance analysis based on metric scores.
        
        TODO: Complete this method to:
        1. Analyze retrieval vs generation performance separately
        2. Calculate average scores for different component types
        3. Identify strongest and weakest areas
        4. Provide component-specific insights
        
        Args:
            results (Dict): Evaluation results with metric scores
        """
        metrics = results["metrics"]
        
        # TODO: Analyze retrieval performance
        retrieval_metrics = ["context_precision", "context_recall", "context_relevance"]
        # Calculate average retrieval score and provide analysis
        retrieval_scores = [metrics.get(m, 0) for m in retrieval_metrics if m in metrics]

        if retrieval_scores:
            avg_retrieval_score = round(np.mean(retrieval_scores), 4)
            print(f"   Average Retrieval Score: {avg_retrieval_score:.4f} -> {'Strong' if avg_retrieval_score >= 0.8 else 'Moderate' if avg_retrieval_score >= 0.6 else 'Weak'} retrieval capabilities")
        
        # TODO: Analyze generation performance
        generation_metrics = ["faithfulness", "answer_relevancy", "answer_correctness"]
        # Calculate average generation score and provide analysis
        generation_scores = [metrics.get(m, 0) for m in generation_metrics if m in metrics]
        
        if generation_scores:
            avg_generation_score = round(np.mean(generation_scores), 4)
            print(f"   Average Generation Score: {avg_generation_score:.4f} -> {'Strong' if avg_generation_score >= 0.8 else 'Moderate' if avg_generation_score >= 0.6 else 'Weak'} generation capabilities")

        # TODO: Identify strengths and weaknesses
        # Find best and worst performing metrics
        if metrics:
            best_metric = max(metrics.items(), key=lambda x: x[1])
            worst_metric = min(metrics.items(), key=lambda x: x[1])
            print(f"   Strongest Metric: {best_metric[0]} ({best_metric[1]:.4f})")
            print(f"   Weakest Metric: {worst_metric[0]} ({worst_metric[1]:.4f})")


    def _provide_recommendations(self, results: Dict) -> None:
        """
        Provide actionable recommendations based on evaluation results.
        
        TODO: Complete this method to:
        1. Generate context-based recommendations for low scores
        2. Provide generation-based recommendations for answer quality
        3. Suggest general improvements based on overall performance
        4. Format recommendations for actionability
        
        Args:
            results (Dict): Evaluation results with metric scores
        """
        metrics = results["metrics"]
        recommendations = []
        
        # TODO: Generate context-based recommendations
        # Check context_precision, context_recall, context_relevancy scores
        
        # TODO: Generate generation-based recommendations
        # Check faithfulness, answer_relevancy, answer_correctness scores
        
        # TODO: Generate general recommendations based on overall score
        
        # TODO: Display recommendations
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")


    def compare_configurations(self, dataset: Dataset, configs: List[str]) -> Dict:
        """
        Compare multiple evaluation configurations on the same dataset.
        
        TODO: Complete this method to:
        1. Evaluate dataset with each specified configuration
        2. Collect and organize comparison results
        3. Create summary comparison with best performers
        4. Handle rate limiting between evaluations
        
        Args:
            dataset (Dataset): RAGAS-compatible evaluation dataset
            configs (List[str]): List of configuration keys to compare
            
        Returns:
            Dict: Comparison results across configurations
        """
        print(f"\n🔄 Comparing {len(configs)} evaluation configurations")
        
        comparison_results = {
            "configurations": configs,
            "results": {},
            "summary": {}
        }
        
        # TODO: Evaluate with each configuration
        # TODO: Create summary comparison
        # TODO: Add delays to avoid rate limiting
        
        return comparison_results


    def generate_evaluation_report(self, results: Dict, dataset_name: str, output_file: str = None) -> str:
        """
        Generate comprehensive evaluation report in markdown format.
        
        TODO: Complete this method to:
        1. Create markdown-formatted report with executive summary
        2. Include detailed metric breakdown and interpretations
        3. Add performance analysis and recommendations
        4. Include technical details and metadata
        5. Save to file if output_file specified
        
        Args:
            results (Dict): Evaluation results
            dataset_name (str): Name of evaluated dataset
            output_file (str): Optional file path to save report
            
        Returns:
            str: Markdown-formatted evaluation report
        """
        # TODO: Create comprehensive markdown report
        report = f"""# RAG System Evaluation Report

            ## Executive Summary
            - **Dataset**: {dataset_name}
            - **Evaluation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            - **Configuration**: {results['config_used']}
            - **Overall Score**: {results['overall_score']:.4f}

            ## Metric Breakdown
            """
        
        # TODO: Add metric details, analysis, and recommendations
        
        # TODO: Save to file if specified
        if output_file:
            # TODO: Write report to file
            pass
        
        return report


# Mock RAG System for demonstration purposes
class MockRAGSystem:
    """
    Mock RAG system for demonstration and testing purposes.
    
    TODO: Complete this class to:
    1. Provide realistic RAG responses for testing
    2. Include performance metrics (timing, context quality)
    3. Support multiple domains and question types
    4. Simulate real RAG system behavior patterns
    """
    
    def __init__(self):
        """Initialize mock RAG system with predefined responses."""
        # TODO: Create realistic response data for different datasets
        self.responses = {
            "technical_qa": [
                # TODO: Add mock responses for technical questions
            ],
            "customer_support": [
                # TODO: Add mock responses for customer support questions
            ]
        }
    

    def get_responses(self, dataset_key: str) -> List[Dict]:
        """
        Get mock RAG responses for a specific dataset.
        
        TODO: Complete this method to:
        1. Return appropriate responses for dataset_key
        2. Handle unknown datasets gracefully
        3. Include realistic response structure and metadata
        
        Args:
            dataset_key (str): Dataset identifier
            
        Returns:
            List[Dict]: Mock RAG responses
        """
        # TODO: Return responses for specified dataset
        return self.responses.get(dataset_key, [])


def demonstrate_ragas_evaluation():
    """
    Comprehensive demonstration of RAGAS evaluation framework.
    
    TODO: Complete this function to:
    1. Initialize evaluator and mock RAG system
    2. Test different datasets and configurations
    3. Display comprehensive results and analysis
    4. Demonstrate configuration comparison
    5. Generate sample evaluation reports
    """
    print("🚀 RAGAS RAG System Evaluation Demonstration")
    print("="*60)
    
    # TODO: Initialize evaluator (replace with your OpenAI API key)
    evaluator = None
    
    # TODO: Initialize mock RAG system
    mock_rag = None
    
    # TODO: Define test scenarios
    test_scenarios = [
        # TODO: Add test scenarios for different datasets and configurations
    ]
    
    # TODO: Run evaluations for each scenario
    # TODO: Display results and analysis
    # TODO: Demonstrate configuration comparison
    # TODO: Generate sample reports


def run_custom_evaluation(dataset_key: str, rag_responses: List[Dict], config_key: str = "comprehensive"):
    """
    Run custom evaluation with user-provided data.
    
    TODO: Complete this function to:
    1. Initialize evaluator with API key
    2. Create evaluation dataset from provided data
    3. Run evaluation with specified configuration
    4. Display comprehensive results
    5. Return evaluation results for further analysis
    
    Args:
        dataset_key (str): Key identifying the test dataset
        rag_responses (List[Dict]): RAG system responses to evaluate
        config_key (str): Evaluation configuration to use
        
    Returns:
        Dict: Evaluation results
    """
    print(f"🔧 Running custom evaluation")
    
    # TODO: Initialize evaluator
    # TODO: Create evaluation dataset
    # TODO: Run evaluation
    # TODO: Display results
    # TODO: Return results


def main():
    ev = RAGSystemEvaluator()


if __name__ == "__main__":
    main()


# TODO: Example usage - uncomment and test when ready
# Run the comprehensive demonstration
# demonstrate_ragas_evaluation()

# TODO: Additional examples you can implement:
#
# Example 1: Custom evaluation with your own RAG system
# custom_responses = [
#     {
#         "question": "Your question here",
#         "answer": "Your RAG system's answer",
#         "contexts": ["Retrieved context 1", "Retrieved context 2"],
#         "retrieval_time": 0.15,
#         "generation_time": 1.2
#     }
# ]
# custom_results = run_custom_evaluation("technical_qa", custom_responses, "comprehensive")
#
# Example 2: Configuration comparison
# evaluator = RAGSystemEvaluator("your_api_key")
# mock_rag = MockRAGSystem()
# responses = mock_rag.get_responses("technical_qa")
# dataset = evaluator.create_evaluation_dataset("technical_qa", responses)
# comparison = evaluator.compare_configurations(dataset, ["comprehensive", "quick_eval"])
#
# Example 3: Generate detailed reports
# report = evaluator.generate_evaluation_report(
#     results, 
#     "Technical Documentation Q&A",
#     "detailed_evaluation_report.md"
# )

"""
EXERCISE COMPLETION CHECKLIST:
✅ Import all necessary libraries (pandas, numpy, ragas, datasets, openai, etc.)
✅ Complete EVALUATION_CONFIGS with appropriate RAGAS metrics for each configuration
✅ Fill in EVALUATION_DATASETS with realistic questions, ground truth, and contexts
✅ Implement RAGSystemEvaluator.__init__() with proper initialization
✅ Complete create_evaluation_dataset() with RAGAS dataset formatting
✅ Implement evaluate_rag_system() with RAGAS evaluation pipeline
✅ Complete display_evaluation_results() with comprehensive result formatting
□ Implement metric interpretation and performance analysis methods
□ Complete compare_configurations() for systematic comparison
□ Implement generate_evaluation_report() for automated documentation
□ Create MockRAGSystem with realistic response data
□ Complete demonstrate_ragas_evaluation() with comprehensive testing
□ Test your implementation with the example usage
□ Add your own OpenAI API key and test the complete workflow

BONUS CHALLENGES:
□ Add custom RAGAS metrics for domain-specific evaluation
□ Implement statistical significance testing for metric comparisons
□ Create visualization dashboards for evaluation results using matplotlib/plotly
□ Add support for batch evaluation of multiple RAG systems
□ Implement automated evaluation pipelines with scheduling
□ Create integration with popular RAG frameworks (LangChain, LlamaIndex)
□ Add support for multi-language evaluation datasets
□ Implement cost tracking and optimization for evaluation processes
□ Create A/B testing framework for RAG system comparison
□ Add real-time evaluation monitoring and alerting systems
"""
