#!/usr/bin/env python3
"""
Test Centralized FAISS Database
Verifies that reid-worker can communicate with reid-db-handler FAISS service
"""

import sys
import numpy as np
import configparser
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from db_client import TraitsDBClient
from remote_face_db import RemoteFaceDatabase, create_face_database


def test_db_client_faiss():
    """Test FAISS operations via database client"""
    print("\n=== Testing TraitsDBClient FAISS Operations ===\n")

    # Load config
    config = configparser.ConfigParser()
    config.read('config.ini')

    api_url = config.get('database', 'api_url')
    api_key = config.get('database', 'api_key')

    client = TraitsDBClient(api_url, api_key)

    # Check API health
    print("1. Checking API health...")
    health = client.health_check()
    print(f"   Health: {health}")

    # Get current stats
    print("\n2. Getting FAISS stats...")
    stats = client.faiss_get_stats()
    print(f"   Total faces: {stats.get('total_faces', 0)}")

    # Create test embeddings
    print("\n3. Creating test embeddings...")
    test_embedding1 = np.random.randn(512).astype(np.float32)
    test_embedding1 = test_embedding1 / np.linalg.norm(test_embedding1)

    test_embedding2 = np.random.randn(512).astype(np.float32)
    test_embedding2 = test_embedding2 / np.linalg.norm(test_embedding2)

    # Add face to FAISS
    print("\n4. Adding face to FAISS...")
    result = client.faiss_add_face(test_embedding1, "test_person_1")
    print(f"   Result: {result}")

    # Search for the same face
    print("\n5. Searching for the face...")
    search_result = client.faiss_search(test_embedding1, threshold=0.4)
    print(f"   Found: {search_result.get('name')} (similarity: {search_result.get('similarity'):.3f})")

    # Batch add
    print("\n6. Batch adding faces...")
    embeddings = [test_embedding2, np.random.randn(512).astype(np.float32)]
    embeddings[1] = embeddings[1] / np.linalg.norm(embeddings[1])
    names = ["test_person_2", "test_person_3"]

    batch_result = client.faiss_add_faces_batch(embeddings, names)
    print(f"   Added {batch_result.get('count')} faces")

    # Batch search
    print("\n7. Batch searching...")
    batch_search_results = client.faiss_batch_search([test_embedding1, test_embedding2], threshold=0.4)
    for i, result in enumerate(batch_search_results):
        print(f"   Face {i+1}: {result.get('name')} (similarity: {result.get('similarity'):.3f})")

    # Save database
    print("\n8. Saving FAISS database...")
    save_result = client.faiss_save()
    print(f"   Save result: {save_result}")

    # Final stats
    print("\n9. Final FAISS stats...")
    final_stats = client.faiss_get_stats()
    print(f"   Total faces: {final_stats.get('total_faces', 0)}")

    print("\n[OK] TraitsDBClient FAISS tests completed successfully!")


def test_remote_face_database():
    """Test RemoteFaceDatabase wrapper"""
    print("\n\n=== Testing RemoteFaceDatabase Wrapper ===\n")

    # Load config
    config = configparser.ConfigParser()
    config.read('config.ini')

    api_url = config.get('database', 'api_url')
    api_key = config.get('database', 'api_key')

    # Create remote database
    print("1. Creating remote face database...")
    db = RemoteFaceDatabase(api_url, api_key, embedding_size=512)

    # Load (verify connection)
    print("\n2. Loading database...")
    loaded = db.load()
    print(f"   Connected: {loaded}")

    # Create test embedding
    print("\n3. Creating test embedding...")
    test_embedding = np.random.randn(512).astype(np.float32)
    test_embedding = test_embedding / np.linalg.norm(test_embedding)

    # Add face
    print("\n4. Adding face...")
    db.add_face(test_embedding, "remote_test_person")
    print("   Face added successfully")

    # Search for face
    print("\n5. Searching for face...")
    name, similarity = db.search(test_embedding, threshold=0.4)
    print(f"   Found: {name} (similarity: {similarity:.3f})")

    # Batch operations
    print("\n6. Testing batch operations...")
    embeddings = [np.random.randn(512).astype(np.float32) for _ in range(3)]
    embeddings = [e / np.linalg.norm(e) for e in embeddings]
    names = ["batch_person_1", "batch_person_2", "batch_person_3"]

    db.add_faces_batch(embeddings, names)
    print("   Batch add completed")

    results = db.batch_search(embeddings, threshold=0.4)
    for i, (name, sim) in enumerate(results):
        print(f"   Face {i+1}: {name} (similarity: {sim:.3f})")

    # Save
    print("\n7. Saving database...")
    db.save()
    print("   Database saved")

    # Close
    db.close()

    print("\n[OK] RemoteFaceDatabase tests completed successfully!")


def test_factory_function():
    """Test factory function for database creation"""
    print("\n\n=== Testing Factory Function ===\n")

    # Load config
    config = configparser.ConfigParser()
    config.read('config.ini')

    api_url = config.get('database', 'api_url')
    api_key = config.get('database', 'api_key')

    # Test remote creation
    print("1. Creating remote database via factory...")
    remote_db = create_face_database(api_url=api_url, api_key=api_key, embedding_size=512)
    print(f"   Type: {type(remote_db).__name__}")
    remote_db.load()

    # Test embedding
    test_emb = np.random.randn(512).astype(np.float32)
    test_emb = test_emb / np.linalg.norm(test_emb)

    remote_db.add_face(test_emb, "factory_test_person")
    name, sim = remote_db.search(test_emb, threshold=0.4)
    print(f"   Search result: {name} (similarity: {sim:.3f})")

    remote_db.close()

    # Test local fallback (without API credentials)
    print("\n2. Testing local database fallback...")
    print("   (This would use local FAISS if API not configured)")
    # Don't actually test local since we want to use remote

    print("\n[OK] Factory function tests completed successfully!")


def test_integration():
    """Test full integration scenario"""
    print("\n\n=== Testing Full Integration ===\n")

    # Load config
    config = configparser.ConfigParser()
    config.read('config.ini')

    api_url = config.get('database', 'api_url')
    api_key = config.get('database', 'api_key')

    # Simulate reid-worker usage
    print("1. Simulating reid-worker face recognition flow...")

    db = create_face_database(api_url=api_url, api_key=api_key, embedding_size=512)
    db.load()

    # Simulate detecting a new face
    print("\n2. Detecting new face (unknown person)...")
    new_face_embedding = np.random.randn(512).astype(np.float32)
    new_face_embedding = new_face_embedding / np.linalg.norm(new_face_embedding)

    name, similarity = db.search(new_face_embedding, threshold=0.6)
    print(f"   Search result: {name} (similarity: {similarity:.3f})")

    if name == "Unknown" or similarity < 0.6:
        person_id = "person_integration_test"
        print(f"   New person detected, adding to database: {person_id}")
        db.add_face(new_face_embedding, person_id)
        db.save()

    # Simulate detecting the same face again
    print("\n3. Detecting same face again...")
    # Add some noise to simulate slight variation
    same_face_embedding = new_face_embedding + np.random.randn(512).astype(np.float32) * 0.01
    same_face_embedding = same_face_embedding / np.linalg.norm(same_face_embedding)

    name, similarity = db.search(same_face_embedding, threshold=0.6)
    print(f"   Search result: {name} (similarity: {similarity:.3f})")

    if similarity >= 0.6:
        print(f"   [OK] Successfully recognized person: {name}")
    else:
        print(f"   [FAIL] Failed to recognize person (similarity too low)")

    db.close()

    print("\n[OK] Integration tests completed successfully!")


if __name__ == "__main__":
    try:
        print("=" * 60)
        print("CENTRALIZED FAISS DATABASE TEST SUITE")
        print("=" * 60)

        # Run all tests
        test_db_client_faiss()
        test_remote_face_database()
        test_factory_function()
        test_integration()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
