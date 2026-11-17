# ChromaDB RAG System Implementation - Starter Template
# TODO: Complete this script to build a Retrieval-Augmented Generation system using ChromaDB

# TODO: Import necessary libraries
# Hint: You'll need chromadb, openai, pandas, time, json, typing, numpy, datetime, uuid, os, pathlib
import chromadb
from chromadb.config import Settings
import openai
from openai import OpenAI
# TODO: Add remaining imports here
import pandas as pd
import time
import json
from typing import Dict, List, Tuple, Optional
import numpy as np
from datetime import datetime
import uuid
import os
from pathlib import Path


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


# TODO: Define embedding configurations for different strategies
# Create configurations for OpenAI embeddings and local alternatives
EMBEDDING_CONFIGS = {
    "openai_embeddings": {
        # TODO: Set provider to "openai"
        "provider": "openai",
        # TODO: Choose OpenAI embedding model (hint: text-embedding-3-small is cost-effective)
        "model": "text-embedding-3-small",
        # TODO: Set dimensions (hint: 1536 for text-embedding-3-small)
        "dimensions": 1536,
        "description": "OpenAI embeddings with excellent semantic understanding"
    },
    "local_embeddings": {
        # TODO: Set provider to "sentence_transformers"
        "provider": "sentence_transformers",
        # TODO: Choose local model (hint: all-MiniLM-L6-v2 is lightweight)
        "model": "all-MiniLM-L6-v2",
        # TODO: Set dimensions (hint: 384 for all-MiniLM-L6-v2)
        "dimensions": 384,
        "description": "Local embeddings for cost-effective processing"
    }
}

# TODO: Define collection configurations for different document types
# Create configurations for technical docs, FAQ support, and knowledge base
COLLECTION_CONFIGS = {
    "tech_docs": {
        # TODO: Set collection name
        "name": "technical_documentation",
        # TODO: Define metadata fields for technical documentation
        "metadata_fields": ["source", "category", "difficulty", "last_updated"],
        "description": "Technical documentation with structured metadata"
    },
    "faq_support": {
        # TODO: Set collection name
        "name": "faq_customer_support",
        # TODO: Define metadata fields for FAQ support
        "metadata_fields": ["category", "source", "priority", "department", "tags"],
        "description": "FAQ database for customer support automation"
    },
    "knowledge_base": {
        # TODO: Set collection name
        "name": "general_knowledge",
        # TODO: Define metadata fields for general knowledge
        "metadata_fields": ["topic", "source", "confidence", "date_added"],
        "description": "General knowledge base for information retrieval"
    }
}

# TODO: Create sample documents for testing
# Define realistic business documents with content and metadata
SAMPLE_DOCUMENTS = {
    "tech_docs": [
        {
            "id": "tech_001",
            # TODO: Add content about ChromaDB (200-300 words)
            "content": "ChromaDB is an open-source vector database built specifically for artificial intelligence and machine learning applications. It provides a simple, developer-friendly way to store, search, and manage high-dimensional vector embeddings, which are the numerical representations of text, images, and audio that power modern AI pipelines. ChromaDB is designed to be lightweight and easy to integrate, making it a popular choice for startups and enterprises that need semantic search, recommendation systems, and retrieval-augmented generation (RAG) without the complexity of traditional database infrastructure. One of ChromaDB's key strengths is its flexibility. It supports multiple embedding functions, letting developers plug in models from OpenAI, sentence-transformers, or their own custom models. It also offers both in-memory storage for quick prototyping and persistent storage for production, where data survives across restarts. ChromaDB natively supports metadata filtering, so users can combine semantic similarity search with structured filters like date ranges, categories, or user IDs. It runs either as an embedded library inside a process or as a standalone server through its Python and HTTP APIs. Its design follows a simple 'three calls' principle (add, update, query) that keeps vector database operations intuitive. Because it stores vectors for fast approximate nearest neighbor (ANN) search, ChromaDB can retrieve the most semantically relevant documents in milliseconds, even among millions of records, making it an ideal backbone for context-aware information retrieval.",
            "metadata": {
                # TODO: Add appropriate metadata
                "source": "ChromaDB Documentation",
                "category": "Database",
                "difficulty": "Intermediate",
                "last_updated": "2024-01-15"
            }
        },
        {
            "id": "tech_002",
            # TODO: Add content about RAG systems (200-300 words)
            "content": "Retrieval-Augmented Generation (RAG) is an AI architecture that combines the knowledge-retrieval strength of traditional search systems with the generative power of large language models (LLMs). Instead of relying only on what is baked into a language model's parameters, a RAG system first retrieves relevant documents or data fragments from an external knowledge base, then feeds that context to the model to produce a grounded, accurate answer. The RAG workflow typically follows four steps. First, a query is converted into a vector embedding using an embedding model. Second, the system runs a similarity search against a vector database to find the most relevant source documents. Third, those documents are formatted into a prompt alongside the original question. Fourth, a large language model generates a response using both the retrieved context and its own training. RAG offers several key advantages. It reduces hallucinations by anchoring responses in real, verifiable sources. It keeps answers current by pulling from up-to-date data without requiring expensive model retraining. It also improves transparency because responses can cite their sources. Common use cases include customer support chatbots, internal enterprise knowledge search, question-answering over documentation, and research assistance. The main engineering challenges are retrieval quality, context-window management, latency, and convincing the model to trust accurate context over misleading priors. When implemented well, RAG turns a generic chatbot into a domain-specific assistant that answers questions with precision and citations.",
            "metadata": {
                # TODO: Add appropriate metadata
                "source": "AI Research Papers",
                "category": "AI",
                "difficulty": "Advanced",
                "last_updated": "2024-02-01"
            }
        },
        {
            "id": "tech_003",
            # TODO: Add content about vector embeddings (200-300 words)
            "content": "Vector embeddings are numerical representations of data - such as text, images, or audio - that capture their semantic meaning in a high-dimensional space. In an embedding, each dimension represents a latent feature of the data, and items with similar meaning sit close together. For example, the embeddings of the words 'king' and 'queen' are much closer than those of 'king' and 'banana', even though both pairs share no literal letters. The process begins by feeding text through an embedding model such as OpenAI's text-embedding-3-small. That neural network maps each piece of text to a fixed-length vector, typically 1536 or 384 numbers long. Similarity between embeddings is commonly measured using cosine similarity or Euclidean (L2) distance: a high cosine similarity means the texts convey related concepts, while a low score means they are unrelated. Embeddings enable capabilities that plain keyword matching cannot achieve. Because they encode meaning rather than exact words, a query like 'my internet is down' can still match a document titled 'How to Troubleshoot Connectivity Issues', even when the words do not overlap. This semantic understanding underpins modern search, deduplication, clustering, and recommendation systems. A critical rule is model consistency: every vector in a single collection must be produced by the same model, because vectors from different models live in incompatible spaces and cannot be compared. Batch-embedding many documents at once also improves efficiency, letting systems process large corpora quickly and cost-effectively.",
            "metadata": {
                # TODO: Add appropriate metadata
                "source": "Embedding Guide",
                "category": "AI",
                "difficulty": "Intermediate",
                "last_updated": "2024-01-20"
            }
        }
    ],
    "faq_support": [
        {
            "id": "faq_001",
            # TODO: Add FAQ about password reset
            "content": "Q: How do I reset my password? A: To reset your password, follow these simple steps. First, go to the login page and click the 'Forgot Password?' link located just below the password field. Next, enter the email address associated with your account and submit the form. You will receive a secure reset link in your inbox - be sure to check your Spam folder if it does not arrive within a few minutes. Click the link, then create a new password that is at least 12 characters long and combines letters, numbers, and symbols. Once you save your new password, you can log in with it. For security, the reset link expires after 24 hours. If you continue to have trouble logging in, or if you no longer have access to your registered email, please contact our IT Support team through the internal help desk, and we will verify your identity and assist you further.",
            "metadata": {
                # TODO: Add appropriate metadata for customer support
                "source": "Customer Support FAQ",
                "category": "Account Management",
                "priority": "High",
                "department": "IT Support",
                "tags": "password, security, login"
            }
        },
        {
            "id": "faq_002",
            # TODO: Add FAQ about business hours
            "content": "Q: What are your business hours? A: Our customer support team is available Monday through Friday from 9 AM to 6 PM EST. During these hours you can reach us by phone, email, or live chat and expect a response within one business day for email inquiries. For urgent technical issues outside regular hours, we offer an emergency support line available 24/7, included at no extra cost with premium subscriptions. Standard plan customers can access the emergency line during business hours and will be routed to the on-call engineer for critical, service-down incidents. We also recommend our self-service knowledge base, which is available around the clock and contains step-by-step guides that often resolve common questions instantly. Support response times may be longer on public holidays, so planning ahead is advised.",
            "metadata": {
                # TODO: Add appropriate metadata
                "source": "Customer Support FAQ",
                "category": "General Information",
                "priority": "Medium",
                "department": "Customer Service",
                "tags": "hours, support, availability"
            }
        },
        {
            "id": "faq_003",
            # TODO: Add FAQ about subscription upgrade
            "content": "Q: How do I upgrade my subscription? A: Upgrading your subscription is quick and easy. Log into your account and navigate to the 'Billing' or 'Plan' section, usually found in account settings or the dashboard menu. From there, browse the available plans and select the tier that best fits your needs. Click 'Upgrade Plan', review the new pricing, and confirm your selection. Your new plan takes effect immediately, and you will be prorated for the remainder of your current billing period, meaning you only pay the difference for the time left in this cycle. Any new features or higher limits are unlocked right away. If you were on an annual plan, the remaining balance is typically credited toward the new total. You will receive a confirmation email and an updated invoice. If you run into any issues during the upgrade process, please contact our Sales team for hands-on assistance.",
            "metadata": {
                # TODO: Add appropriate metadata
                "source": "Sales & Billing FAQ",
                "category": "Billing",
                "priority": "High",
                "department": "Sales",
                "tags": "subscription, billing, upgrade"
            }
        }
    ]
}

class ChromaDBRAGSystem:
    """
    A comprehensive RAG system implementation using ChromaDB for vector storage and retrieval.
    
    TODO: Complete this class to implement a production-ready RAG system with:
    - ChromaDB integration for vector storage
    - Embedding generation and management
    - Document ingestion and retrieval
    - RAG response generation
    """
    
    def __init__(self, embedding_config: str = "openai_embeddings", persist_directory: str = "./chroma_db"):
        """
        Initialize the ChromaDB RAG system with specified configuration.
        
        TODO: Complete this method to:
        1. Store configuration parameters
        2. Initialize ChromaDB client with persistent storage
        3. Initialize OpenAI client for embeddings and generation
        4. Set up collections dictionary
        5. Print initialization status
        
        Args:
            embedding_config (str): Configuration key for embedding strategy
            persist_directory (str): Directory for persistent storage
        """
        # TODO: Store embedding configuration
        self.embedding_config = EMBEDDING_CONFIGS.get(embedding_config)
        self.persist_directory = persist_directory
        
        # TODO: Initialize ChromaDB client with persistent storage
        # Hint: Use chromadb.PersistentClient with path and settings
        self.client = chromadb.PersistentClient(
            path=self.persist_directory, 
            settings=Settings(
                anonymized_telemetry=False,  # Disable telemetry for privacy
                allow_reset=True  # Allow resetting collections for development
            )
        )
        
        # TODO: Initialize OpenAI client for embeddings and generation
        # SECURITY NOTE: Use environment variables for API keys in production
        self.openai_client = get_client()

        # TODO: Initialize collections dictionary
        self.collections = {}
        
        # TODO: Print initialization status
        print(f"🚀 ChromaDB RAG System initialized")
        print(f"   Embedding Config: {self.embedding_config['description']}")
        print(f"   Persistent Storage Directory: {self.persist_directory}")
        print(f"   OpenAI Client Initialized: {'Yes' if self.openai_client else 'No'}")
        print(f"   Collections Ready: {len(self.collections)} (create collections to start)")


    def create_collection(self, collection_key: str) -> chromadb.Collection:
        """
        Create a new ChromaDB collection with specified configuration.
        
        TODO: Complete this method to:
        1. Validate collection_key exists in COLLECTION_CONFIGS
        2. Get collection configuration
        3. Delete existing collection if it exists (for development)
        4. Create new collection with appropriate settings
        5. Store collection in self.collections
        6. Handle errors gracefully
        
        Args:
            collection_key (str): Key from COLLECTION_CONFIGS
            
        Returns:
            chromadb.Collection: The created collection object
        """
        # TODO: Validate collection_key
        if collection_key not in COLLECTION_CONFIGS:
            raise ValueError(f"Unknown collection configuration: {collection_key}")
            
        # TODO: Get configuration and create collection
        config = COLLECTION_CONFIGS[collection_key]
        collection_name = config["name"]
        
        print(f"\n📁 Creating collection: {collection_name}")
        
        try:
            # TODO: Delete existing collection if it exists
            try:
                self.client.delete_collection(collection_name)
                print(f"   ♻️  Deleted existing collection")
            except Exception:
                pass  # Collection does not exist yet — safe to ignore

            # TODO: Create new collection
            collection = self.client.create_collection(
                name=collection_name,
                embedding_function=None,  # We'll handle embeddings manually
                metadata={"description": config["description"]}
            )
            # TODO: Store in self.collections
            self.collections[collection_key] = collection
            # TODO: Print success message
            print(f"   ✅ Collection '{collection_name}' created successfully")

            return collection # returning due to docstring spec
                                 
        except Exception as e:
            print(f"   ❌ Error creating collection: {str(e)}")
            raise


    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using the configured embedding model.
        
        TODO: Complete this method to:
        1. Handle OpenAI embedding generation
        2. Support local embedding generation (optional)
        3. Include proper error handling
        4. Return list of embedding vectors
        
        Args:
            texts (List[str]): List of texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        print(f"🔄 Generating embeddings for {len(texts)} texts...")
        
        try:
            if self.embedding_config["provider"] == "openai":
                # TODO: Use OpenAI embeddings API
                # Hint: Use self.openai_client.embeddings.create()
                response = self.openai_client.embeddings.create(
                    model=self.embedding_config["model"],
                    input=texts
                )
                embeddings = [item.embedding for item in response.data]
                print(f"   ✅ Generated {len(embeddings)} OpenAI embeddings successfully")
                return embeddings
                
            else:
                # TODO: Handle local embeddings (optional)
                print("⚠️  Local embeddings not implemented in this example")
                return []
                
        except Exception as e:
            print(f"❌ Error generating embeddings: {str(e)}")
            raise

    def add_documents(self, collection_key: str, documents: List[Dict]) -> None:
        """
        Add documents to a ChromaDB collection with embeddings and metadata.
        
        TODO: Complete this method to:
        1. Validate collection exists
        2. Extract texts, IDs, and metadata from documents
        3. Generate embeddings for texts
        4. Add documents to collection with embeddings
        5. Handle errors and provide status updates
        
        Args:
            collection_key (str): Key identifying the target collection
            documents (List[Dict]): List of document dictionaries with content and metadata
        """
        # TODO: Validate collection exists
        if collection_key not in self.collections:
            raise ValueError(f"Collection {collection_key} not found. Create it first.")
            
        # TODO: Get collection and extract document data
        collection = self.collections[collection_key]
        
        print(f"\n📄 Adding {len(documents)} documents to {collection.name}")
        
        # TODO: Extract texts, IDs, and metadata
        texts = [doc["content"] for doc in documents]
        ids = [doc["id"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]

        # TODO: Generate embeddings
        embeddings = self.generate_embeddings(texts)

        if not embeddings:
            print("   ❌ No embeddings generated; documents not added.")
            return

        # TODO: Add documents to collection
        # TODO: Print success status
        try:
            collection.add(
                documents=texts,
                ids=ids,
                metadatas=metadatas,
                embeddings=embeddings
            )
            print(f"   ✅ Successfully added {len(documents)} documents to {collection.name}")
            print(f"   Collection now contains {collection.count()} documents")
        except Exception as e:
            print(f"   ❌ Error adding documents: {str(e)}")
            raise
    

    def search_documents(self, collection_key: str, query: str, n_results: int = 3, 
                        metadata_filter: Optional[Dict] = None) -> Dict:
        """
        Search for relevant documents using semantic similarity.
        
        TODO: Complete this method to:
        1. Validate collection exists
        2. Generate embedding for query
        3. Perform similarity search with optional metadata filtering
        4. Format and return results with similarity scores
        
        Args:
            collection_key (str): Key identifying the collection to search
            query (str): Search query text
            n_results (int): Number of results to return
            metadata_filter (Optional[Dict]): Metadata filters to apply
            
        Returns:
            Dict: Search results with documents, distances, and metadata
        """
        # TODO: Validate collection exists
        if collection_key not in self.collections:
            raise ValueError(f"Collection {collection_key} not found.")

        collection = self.collections[collection_key]
        print(f"\n🔍 Searching collection: {collection.name}")
        print(f"   Query: '{query}'")
        print(f"   Number of results requested: {n_results}")

        # TODO: Generate query embedding
        try:
            query_embedding = self.generate_embeddings([query])

            if not query_embedding:
                print("   ❌ No embedding generated for query; search aborted.")
                return {
                    "query": query,
                    "results": [],
                    "n_results": 0
                }

        except Exception as e:
            print(f"   ❌ Error generating query embedding: {str(e)}")
            return {
                "query": query,
                "results": [],
                "n_results": 0
            }
        
        # TODO: Perform similarity search
        try:
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                where=metadata_filter or None,
                include=["documents", "metadatas", "distances"]
            )

            print(f"   ✅ Retrieved {len(results['documents'][0])} results")
        except Exception as e:
            print(f"   ❌ Error performing similarity search: {str(e)}")
            return {
                "query": query,
                "results": [],
                "n_results": 0
            }

        # TODO: Format and return results
        try:
            formatted_results = {
                "query": query,
                "results": [],
                "n_results": len(results["documents"][0])
            }

            for i in range(len(results["documents"][0])):
                formatted_results["results"].append({
                    "document": results["documents"][0][i],
                    "similarity": 1 - results["distances"][0][i],  # Convert distance to similarity
                    "metadata": results["metadatas"][0][i]
                })

            return formatted_results
        except Exception as e:
            print(f"   ❌ Error formatting search results: {str(e)}")
            return {
                "query": query,
                "results": [],
                "n_results": 0
            }
        
    
    def generate_rag_response(self, collection_key: str, query: str, n_context: int = 3,
                            model: str = "gpt-4o-mini") -> Dict:
        """
        Generate a response using Retrieval-Augmented Generation.
        
        TODO: Complete this method to:
        1. Retrieve relevant context documents
        2. Prepare context for generation
        3. Create prompt with context
        4. Generate response using OpenAI
        5. Format comprehensive response with metadata
        
        Args:
            collection_key (str): Collection to search for context
            query (str): User query to answer
            n_context (int): Number of context documents to retrieve
            model (str): OpenAI model to use for generation
            
        Returns:
            Dict: RAG response with context, answer, and metadata
        """
        print(f"\n🤖 Generating RAG response")
        print(f"   Query: '{query}'")
        
        # TODO: Retrieve relevant context
        search_results = self.search_documents(collection_key, query, n_results=n_context)
        context_docs = search_results.get("results", [])

        if not context_docs:
            print("   ⚠️  No context documents found; proceeding without context")
            return {
                "query": query,
                "context": [],
                "answer": "No relevant context found to generate an answer.",
                "metadata": {"model": model, "sources": []}
            }
        
        # TODO: Prepare context for generation
        context_texts = "\n\n".join(f"[Doc {i + 1}] {doc['document']}" for i, doc in enumerate(context_docs))

        # TODO: Create prompt with context
        system_prompt = ("You are a helpful assistant. Answer the user's question using ONLY the provided context below. If the answer is not contained within the context, respond with 'I don't know.'\n\n")
        user_prompt = f"Context:\n{context_texts}\n\nQuestion: {query}\n\nAnswer:"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # TODO: Generate response using OpenAI
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
            )

            answer = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"   ❌ Error generating RAG response: {str(e)}")
            raise

        # TODO: Format and return comprehensive response
        formatted_response = {
            "query": query,
            "answer": answer,
            "context": context_docs,
            "metadata": {
                "model": model,
                "sources": [doc["metadata"] for doc in context_docs],
                "n_sources": len(context_docs)
            }
        }

        print(f"   ✅ Response generated using model: {model}")
        print(f"   Sources used: {len(context_docs)}")
        return formatted_response
    

    def display_rag_response(self, rag_response: Dict) -> None:
        """
        Display RAG response in a formatted, readable way.
        
        TODO: Complete this method to:
        1. Display question and answer clearly
        2. Show context sources with similarity scores
        3. Display performance metrics
        4. Format output for readability
        
        Args:
            rag_response (Dict): RAG response dictionary from generate_rag_response
        """
        # TODO: Format and display RAG response
        # Include: question, answer, context sources, performance metrics
        print("\n" + "=" * 80)
        print("📖 RAG Response")
        print("=" * 80)
        print(f"\n❓ Question:")
        print(f"    {rag_response.get('query', 'N/A')}")
        print(f"\n💡 Answer:")
        print(f"    {rag_response.get('answer', 'N/A')}")

        context_docs = rag_response.get("context", [])
        if context_docs:
            print(f"\n📚 Context Sources ({len(context_docs)}) documents:")
            for i, doc in enumerate(context_docs):
                source = doc.get("metadata", {}).get("source", "N/A")
                print(f"   {i + 1}. Source: {source}")
                print(f"      Preview: {doc['document'][:120]}...") # Show first 120 chars
        else:
            print("\n📚 No context sources were used for this answer.")

        metadata = rag_response.get("metadata", {})
        print(f"\n⚙️  Metadata:")
        print(f"   Model Used: {metadata.get('model', 'N/A')}")
        print(f"   Number of Sources: {metadata.get('n_sources', len(context_docs))}")
        print("=" * 80)
    

def demonstrate_chromadb_rag():
    """
    Comprehensive demonstration of ChromaDB RAG system capabilities.
    
    TODO: Complete this function to:
    1. Initialize the RAG system
    2. Create collections for different document types
    3. Add sample documents to collections
    4. Test various query types
    5. Display results and performance metrics
    """
    print("🚀 ChromaDB RAG System Demonstration")
    print("="*60)
    
    # TODO: Initialize the RAG system
    rag_system = ChromaDBRAGSystem(embedding_config="openai_embeddings", persist_directory="./demo_chroma_db")

    # TODO: Create collections
    print("\n📁 Creating collections...")
    rag_system.create_collection("tech_docs")
    rag_system.create_collection("faq_support")

    # TODO: Add sample documents
    print("\n📄 Adding sample documents to collections...")
    rag_system.add_documents("tech_docs", SAMPLE_DOCUMENTS["tech_docs"])
    rag_system.add_documents("faq_support", SAMPLE_DOCUMENTS["faq_support"])

    # TODO: Test various queries
    test_queries = [
        {
            "collection": "tech_docs",
            "query": "What is ChromaDB and how does it work?",
            "description": "Technical documentation query"
        },
        {
            "collection": "faq_support",
            "query": "How can I reset my password?",
            "description": "Customer support query"
        },
        {
           "collection": "faq_support",
           "query": "What are your business hours?",
           "description": "General information query"
        }
    ]

    # TODO: Display results
    print("\n🔍 Testing RAG System with various queries...")
    for i, test in enumerate(test_queries, 1):
        print(f"\n{'=' * 20} TEST QUERY {i}: {test['description']} {'=' * 20}")
        rag_response = rag_system.generate_rag_response(
            collection_key=test["collection"], 
            query=test["query"],
            n_context=2
        )
        rag_system.display_rag_response(rag_response)

    # Summary of demonstration
    total_docs = sum(len(docs) for docs in SAMPLE_DOCUMENTS.values())
    print(f"\n🎉 ChromaDB RAG demonstration completed successfully!")
    print(f"   Collections created: {len(rag_system.collections)}")
    print(f"   Total documents added: {total_docs}")
    print(f"   Queries tested: {len(test_queries)}")


def main():
    """
    Main function to run the ChromaDB RAG system demonstration.
    """
    # TODO: Example usage - uncomment and test when ready
    # Run the comprehensive demonstration
    demonstrate_chromadb_rag()

    # TODO: Additional examples you can implement:
    # 
    # Example 1: Custom metadata filtering
    # rag_system = ChromaDBRAGSystem()
    # rag_system.create_collection("tech_docs")
    # rag_system.add_documents("tech_docs", SAMPLE_DOCUMENTS["tech_docs"])
    # filtered_results = rag_system.search_documents(
    #     "tech_docs", 
    #     "database information",
    #     metadata_filter={"category": "Database"}
    # )
    #
    # Example 2: Batch processing multiple queries
    # queries = ["What is RAG?", "How do embeddings work?", "ChromaDB features"]
    # for query in queries:
    #     response = rag_system.generate_rag_response("tech_docs", query)
    #     rag_system.display_rag_response(response)


if __name__ == "__main__":
    main()


"""
EXERCISE COMPLETION CHECKLIST:
✅ Import all necessary libraries
✅ Complete EMBEDDING_CONFIGS with appropriate models and parameters
✅ Fill in COLLECTION_CONFIGS with meaningful names and metadata fields
✅ Create comprehensive SAMPLE_DOCUMENTS with realistic content
✅ Implement ChromaDBRAGSystem.__init__() with proper initialization
✅ Complete create_collection() with ChromaDB collection creation
✅ Implement generate_embeddings() with OpenAI API integration
✅ Complete add_documents() with embedding generation and storage
✅ Implement search_documents() with similarity search and filtering
✅ Complete generate_rag_response() with full RAG pipeline
✅ Implement display_rag_response() with formatted output
✅ Complete demonstrate_chromadb_rag() with comprehensive testing
✅ Test your implementation with the example usage
✅ Add your own API key and test the complete workflow

BONUS CHALLENGES:
□ Add support for local embedding models using sentence-transformers
□ Implement batch processing for large document collections
□ Add metadata-based filtering and advanced search capabilities
□ Create a web interface for the RAG system using Flask or FastAPI
□ Implement document update and deletion functionality
□ Add support for different file formats (PDF, Word, etc.)
□ Create performance monitoring and analytics dashboard
□ Implement user authentication and access control
□ Add support for multi-modal documents (text + images)
□ Create automated document ingestion from external sources
"""
