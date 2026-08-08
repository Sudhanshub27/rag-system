"""
One-Time Data Migration & Cleanup Script for Data Isolation

Inspects legacy global ChromaDB collections ('documents') and either:
1. Purges mixed/unisolated legacy data to ensure clean tenant boundaries.
2. Migrates legacy unisolated data into a designated user collection ('user_legacy').

Usage:
    python scripts/migrate_isolation.py --action purge
    python scripts/migrate_isolation.py --action migrate --target-user legacy_user
"""

import argparse
import sys
from pathlib import Path

import chromadb

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import vector_store_config
from utils.helpers import sanitize_collection_name
from utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Tenant Data Isolation Migration & Cleanup Tool"
    )
    parser.add_argument(
        "--action",
        choices=["purge", "migrate", "inspect"],
        default="inspect",
        help="Action to perform on legacy unisolated collection: inspect, purge, or migrate",
    )
    parser.add_argument(
        "--target-user",
        default="legacy_user",
        help="Target user_id when migrating legacy collection (default: legacy_user)",
    )

    args = parser.parse_args()
    setup_logger()

    persist_dir = vector_store_config.persist_directory
    client = chromadb.PersistentClient(path=persist_dir)

    collections = client.list_collections()
    col_names = [c.name for c in collections]

    print("\n" + "=" * 70)
    print("CHROMADB COLLECTION ISOLATION REPORT")
    print("=" * 70)
    print(f"Persist Directory: {persist_dir}")
    print(f"Total Collections Found: {len(col_names)}")
    for name in col_names:
        c = client.get_collection(name)
        print(f"  • Collection: '{name}' | Chunks: {c.count()}")
    print("=" * 70 + "\n")

    legacy_name = vector_store_config.collection_name  # 'documents'
    if legacy_name not in col_names:
        print(
            f"✅ No legacy unisolated collection ('{legacy_name}') found. Database is clean!"
        )
        return

    legacy_col = client.get_collection(legacy_name)
    legacy_count = legacy_col.count()

    if legacy_count == 0:
        print(
            f"✅ Legacy collection '{legacy_name}' is empty. Deleting empty legacy container."
        )
        client.delete_collection(legacy_name)
        return

    if args.action == "inspect":
        print(
            f"⚠️ Legacy collection '{legacy_name}' contains {legacy_count} unisolated chunks.\n"
            f"Run with '--action purge' to clear legacy mixed data or '--action migrate' to copy into user collection."
        )

    elif args.action == "purge":
        print(
            f"🗑️ Purging {legacy_count} unisolated legacy chunks from collection '{legacy_name}'..."
        )
        client.delete_collection(legacy_name)
        print("✅ Legacy collection purged successfully.")

    elif args.action == "migrate":
        target_col_name = sanitize_collection_name(args.target_user)
        print(
            f"🚚 Migrating {legacy_count} legacy chunks into isolated collection '{target_col_name}' (user_id='{args.target_user}')..."
        )
        target_col = client.get_or_create_collection(
            name=target_col_name,
            metadata={"hnsw:space": "cosine"},
        )

        records = legacy_col.get(include=["documents", "metadatas", "embeddings"])
        if records and records.get("ids"):
            target_col.upsert(
                ids=records["ids"],
                embeddings=records["embeddings"],
                documents=records["documents"],
                metadatas=records["metadatas"],
            )
            print(f"✅ Migrated {len(records['ids'])} chunks to '{target_col_name}'.")

        print(f"Deleting old unisolated collection '{legacy_name}'...")
        client.delete_collection(legacy_name)
        print("✅ Migration complete!")


if __name__ == "__main__":
    main()
