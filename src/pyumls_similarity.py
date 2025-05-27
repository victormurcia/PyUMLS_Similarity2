import mysql.connector
import networkx as nx
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import math
import gensim.downloader as api
from sentence_transformers import SentenceTransformer
import numpy as np
import re
from pyvis.network import Network
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import traceback
from neo4j import GraphDatabase
from itertools import groupby
from collections import OrderedDict
from datasets import load_dataset
from collections import Counter
import json
from nltk.util import ngrams
from nltk.corpus import stopwords
import ijson
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import matplotlib.cm as cm

class UMLSSemanticEngine:

    custom_palette = [
        "#52357B",
        "#5459AC",
        "#648DB3",
        "#B2D8CE",
        "#0E2148",
        "#E3D095",
        "#0E2148",
        "#C62300",
        "#006A67",
        "#1F6E8C"
    ]

    # === Initialization & MySQL Connection ===
    def __init__(self, mysql_info):
        self.mysql_info = mysql_info
        self.connection = None
        self.cursor = None
        self.connect()

    def connect(self):
        """
        Establish connection to the MySQL database.
        Checks for required connection parameters and handles connection errors.
        Prints summary diagnostics if successful.
        """

        ascii_art = r"""
        ╔════════════════════════════════════════════════════════╗
        ║                    PYUMLS SIMILARITY                   ║
        ║            Unified Medical Language System Tools       ║
        ╚════════════════════════════════════════════════════════╝
        """
        print(ascii_art)

        # 1. Validate connection parameters
        required_keys = {"host", "user", "password", "database"}
        missing_keys = required_keys - self.mysql_info.keys()
        if missing_keys:
            raise ValueError(f"Missing required MySQL connection parameters: {missing_keys}")

        # 2. Attempt connection
        try:
            self.connection = mysql.connector.connect(**self.mysql_info)
            self.cursor = self.connection.cursor()
            print("✅ Successfully connected to the database.")
            self.print_connection_summary()

        except mysql.connector.Error as err:
            print(f"❌ MySQL connection error: {err}")
            self.connection = None
            self.cursor = None

    def print_connection_summary(self, get_table_row_counts=False):
        """
        Prints diagnostic information about the current MySQL database connection,
        including server version, user, schema stats, charset, vocabulary presence,
        UMLS release metadata, and term counts by source.
        Parameters:
        - get_table_row_counts (bool): If True, includes row counts for core UMLS tables.
        """
        if not self.connection or not self.cursor:
            print("❌ No active MySQL connection. Please call `connect()` first.")
            return

        try:
            # Server info
            server_info = self.connection.server_info
            user = self.mysql_info.get("user", "N/A")
            host = self.mysql_info.get("host", "N/A")
            db = self.mysql_info.get("database", "N/A")
            print("🔗 MySQL Connection Summary")
            print(f"   • User: {user}")
            print(f"   • Host: {host}")
            print(f"   • Database: {db}")
            print(f"   • Server Version: {server_info}")

            # Charset and collation
            self.cursor.execute("SELECT @@character_set_database, @@collation_database;")
            charset, collation = self.cursor.fetchone()
            print(f"   • Charset: {charset}, Collation: {collation}")

            # Tables
            self.cursor.execute("SHOW TABLES;")
            tables = [row[0] for row in self.cursor.fetchall()]
            print(f"   • Tables ({len(tables)}): {', '.join(tables[:10])}{'...' if len(tables) > 10 else ''}")

            # Normalize for comparison
            existing_tables = {t.upper() for t in tables}
            umls_expected = {"MRCONSO", "MRREL", "MRSTY", "MRHIER", "MRDEF"}
            missing = umls_expected - existing_tables

            if missing:
                print(f"   ⚠️ Missing expected UMLS tables: {', '.join(sorted(missing))}")
            else:
                print("   ✅ All core UMLS tables [MRCONSO", "MRREL", "MRSTY", "MRHIER", "MRDEF] detected.")

            # Row counts for core tables (takes a bit...)
            if get_table_row_counts:
                sample_tables = sorted(umls_expected & existing_tables)
                for table in sample_tables:
                    try:
                        self.cursor.execute(f"SELECT COUNT(*) FROM {table};")
                        count = self.cursor.fetchone()[0]
                        print(f"   • {table}: {count:,} rows")
                    except Exception:
                        print(f"   ⚠️ Could not retrieve row count for {table}")

            # UMLS release date (from MRDOC)
            if "MRDOC" in existing_tables:
                try:
                    self.cursor.execute("SELECT VALUE FROM MRDOC WHERE DOCKEY = 'release_date' LIMIT 1;")
                    release = self.cursor.fetchone()
                    if release:
                        print(f"   • UMLS Release Date: {release[0]}")
                except Exception:
                    print("   ⚠️ Could not retrieve UMLS release date from MRDOC.")

            # Available SABs
            if "MRCONSO" in existing_tables:
                try:
                    self.cursor.execute("SELECT DISTINCT SAB FROM MRCONSO;")
                    sabs = sorted({row[0] for row in self.cursor.fetchall()})
                    print(f"   • Vocabularies (SABs) available: {', '.join(sabs[:10])}{'...' if len(sabs) > 10 else ''}")

                    # Show counts per SAB
                    self.cursor.execute("""
                        SELECT SAB, COUNT(*) as count 
                        FROM MRCONSO 
                        GROUP BY SAB 
                        ORDER BY count DESC 
                        LIMIT 5;
                    """)
                    print("   • Top SABs by concept count:")
                    for sab, count in self.cursor.fetchall():
                        print(f"     - {sab}: {count:,} entries")
                except Exception:
                    print("   ⚠️ Could not retrieve SAB statistics from MRCONSO.")

        except Exception as e:
            print(f"❌ Error during connection summary: {e}")

    def close(self):
        """
        Close the MySQL connection and cursor safely.
        """
        # Close cursor if it exists and is open
        if hasattr(self, 'cursor') and self.cursor:
            try:
                self.cursor.close()
                print("✅ Cursor closed.")
            except Exception as e:
                print(f"⚠️ Error closing cursor: {e}")
            finally:
                self.cursor = None

        # Close connection if it exists and is open
        if hasattr(self, 'connection') and self.connection:
            try:
                self.connection.close()
                print("✅ Connection closed.")
            except Exception as e:
                print(f"⚠️ Error closing connection: {e}")
            finally:
                self.connection = None

    # === UMLS Term Resolution ===
    def find_cui(self, term, sab=None, lat="ENG", verbose=False):
        """
        Find the CUI for a given term from one or more source vocabularies.
        
        Parameters:
        - term (str): The term to search for
        - sab (str or list of str or None): Source vocab(s) to search (e.g., "MSH", ["MSH", "SNOMEDCT_US"])
        - lat (str): Language (default: "ENG")
        
        Returns:
        - CUI (str) if found, else None
        """
        def vprint(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)

        if not term:
            vprint("⚠️ Empty or invalid term")
            return None

        if not hasattr(self, 'cursor') or self.cursor is None:
            vprint("⚠️ No database cursor available. Are you connected?")
            return None

        sabs_to_try = []
        if sab is None:
            sabs_to_try = ["MSH"]
        elif isinstance(sab, str):
            sabs_to_try = [sab]
        else:
            sabs_to_try = list(sab)

        try:
            for current_sab in sabs_to_try:
                query = """
                SELECT CUI FROM MRCONSO
                WHERE STR = %s AND LAT = %s AND SAB = %s
                LIMIT 1
                """
                self.cursor.execute(query, (term, lat, current_sab))
                result = self.cursor.fetchone()
                while self.cursor.nextset():  # Clean up any buffered results
                    pass
                if result:
                    return result[0]

            return None  # No match found

        except mysql.connector.errors.InterfaceError as err:
            vprint(f"⚠️ MySQL interface error for term '{term}': {err}")
            self.connection.rollback()
            return None
        except mysql.connector.Error as err:
            vprint(f"⚠️ Database error when finding CUI for term '{term}': {err}")
            self.connection.rollback()
            return None
        except Exception as e:
            vprint(f"⚠️ Unexpected error in find_cui: {e}")
            return None

    def get_synonyms(self, cui: str, sab_filter: str = None, tty_filter: list = None, lang: str = 'ENG') -> list:
        """
        Retrieve synonymous terms (STRs) for a given CUI from the MRCONSO table.

        Parameters:
        - cui (str): The CUI to query.
        - sab_filter (str, optional): Restrict to a specific source vocabulary (e.g., "MSH", "SNOMEDCT_US").
        - tty_filter (list of str, optional): Restrict to specific term types (e.g., ["PT", "SY"]).
        - lang (str, optional): Language code (default: "ENG").

        Returns:
        - List of dictionaries: [{"str": ..., "sab": ..., "tty": ...}, ...]
        """
        if not self.cursor:
            print("❌ No active database connection.")
            return []

        query = "SELECT STR, SAB, TTY FROM MRCONSO WHERE CUI = %s AND LAT = %s"
        params = [cui, lang]

        if sab_filter:
            query += " AND SAB = %s"
            params.append(sab_filter)

        if tty_filter:
            placeholders = ','.join(['%s'] * len(tty_filter))
            query += f" AND TTY IN ({placeholders})"
            params.extend(tty_filter)

        try:
            self.cursor.execute(query, tuple(params))
            rows = self.cursor.fetchall()
            return [{"str": row[0], "sab": row[1], "tty": row[2]} for row in rows]
        except Exception as e:
            print(f"⚠️ Error retrieving synonyms for CUI {cui}: {e}")
            return []

    def concept_exists_in_ontology(self, cui, sab):
        """
        Check if a CUI exists in the ontology for the given source vocabulary.
        Returns True if it exists, else False.
        """
        if not cui or not sab:
            print(f"⚠️ Invalid input: cui='{cui}', sab='{sab}'")
            return False

        if not hasattr(self, 'cursor') or self.cursor is None:
            print("⚠️ No database cursor available. Are you connected?")
            return False

        query = """
        SELECT 1 FROM MRCONSO
        WHERE CUI = %s AND SAB = %s
        LIMIT 1
        """
        try:
            self.cursor.execute(query, (cui, sab))
            result = self.cursor.fetchone()
            return result is not None
        except mysql.connector.Error as err:
            print(f"⚠️ Database error when checking CUI '{cui}': {err}")
            return False
        except Exception as e:
            print(f"⚠️ Unexpected error in concept_exists_in_ontology: {e}")
            return False

    def get_str(self, cui, sab):
        """
        Retrieve the STR (string) for a CUI, prioritizing preferred forms.
        Falls back gracefully if preferred not found.
        """
        if not cui or not sab:
            print(f"⚠️ Invalid input to get_str: cui='{cui}', sab='{sab}'")
            return "UNKNOWN"

        if not hasattr(self, 'cursor') or self.cursor is None:
            print("⚠️ No database cursor available. Are you connected?")
            return "UNKNOWN"

        query_strict = """
        SELECT STR FROM MRCONSO
        WHERE CUI = %s AND SAB = %s AND TS = 'P' AND STT = 'PF'
        LIMIT 1
        """
        query_fallback = """
        SELECT STR FROM MRCONSO
        WHERE CUI = %s AND SAB = %s
        LIMIT 1
        """

        try:
            # First, try strict match
            self.cursor.execute(query_strict, (cui, sab))
            result = self.cursor.fetchone()
            if result:
                return result[0]
            
            # If strict match not found, try relaxed match
            self.cursor.execute(query_fallback, (cui, sab))
            result = self.cursor.fetchone()
            return result[0] if result else "UNKNOWN"
        except mysql.connector.Error as err:
            print(f"⚠️ Database error in get_str for CUI '{cui}': {err}")
            return "UNKNOWN"
        except Exception as e:
            print(f"⚠️ Unexpected error in get_str for CUI '{cui}': {e}")
            return "UNKNOWN"

    def get_def(self, cui, sab):
        """
        Retrieve the DEF (definition) for a given CUI and source vocabulary (SAB).
        Falls back gracefully if not found.
        """
        if not cui or not sab:
            print(f"⚠️ Invalid input to get_def: cui='{cui}', sab='{sab}'")
            return "No definition found."

        if not hasattr(self, 'cursor') or self.cursor is None:
            print("⚠️ No database cursor available. Are you connected?")
            return "No definition found."

        query = """
        SELECT DEF FROM MRDEF
        WHERE CUI = %s AND SAB = %s
        LIMIT 1
        """

        try:
            self.cursor.execute(query, (cui, sab))
            result = self.cursor.fetchone()
            return result[0] if result else "No definition found."
        except mysql.connector.Error as err:
            print(f"⚠️ Database error in get_def for CUI '{cui}': {err}")
            return "No definition found."
        except Exception as e:
            print(f"⚠️ Unexpected error in get_def for CUI '{cui}': {e}")
            return "No definition found."    

    def resolve_cui(self, input_val, sab="MSH"):
        """
        Helper to resolve either a CUI or a term into a CUI.
        """
        if input_val.upper().startswith('C') and input_val[1:].isdigit():
            return input_val  # Already a CUI
        else:
            return self.find_cui(input_val, sab)
      
    # === Semantic Types & Groups ===
    def get_semantic_types(self, cui):
        """
        Retrieve all semantic types (TUI + STY) associated with a CUI.
        (No SAB filtering — MRSTY does not have SAB.)
        """
        query = """
        SELECT DISTINCT TUI, STY FROM MRSTY
        WHERE CUI = %s
        """
        self.cursor.execute(query, (cui,))
        results = self.cursor.fetchall()
        if results:
            return [{"TUI": tui, "STY": sty} for tui, sty in results]
        else:
            return []

    def get_semantic_type_definition(self, tui):
        """
        Retrieve the definition of a semantic type (TUI) from SRDEF table.
        """
        query = """
        SELECT DEF FROM SRDEF
        WHERE UI = %s
        """
        self.cursor.execute(query, (tui,))
        result = self.cursor.fetchone()
        return result[0] if result else "Definition not found."

    def get_semantic_group(self, tui):
        """
        Retrieve the semantic group name for a given TUI using SRSTRE1.
        """
        query = """
        SELECT SG FROM SRSTRE1
        WHERE TUI = %s
        """
        self.cursor.execute(query, (tui,))
        result = self.cursor.fetchone()
        return result[0] if result else "Group not found."

    def load_semantic_group_mapping(self,filepath):
        """
        Loads a mapping from TUI to semantic group name from SemGroups.txt file.
        
        Parameters:
        - filepath (str): Path to the SemGroups.txt file
        
        Returns:
        - dict: {TUI: Semantic Group Name}
        """
        mapping = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 3:
                    group_name = parts[1]
                    tui = parts[2]
                    mapping[tui] = group_name
        return mapping
    
    def get_semantic_groups_by_cui(self, cui, tui_to_group):
        """
        Given a CUI, return a list of semantic groups it belongs to.

        Parameters:
        - cui (str): UMLS Concept Unique Identifier
        - tui_to_group (dict): Mapping of TUI to semantic group name
        
        Returns:
        - list of semantic group names
        """
        query = "SELECT DISTINCT TUI FROM MRSTY WHERE CUI = %s"
        self.cursor.execute(query, (cui,))
        tuis = [row[0] for row in self.cursor.fetchall()]
        
        groups = {tui_to_group.get(tui, "Unknown") for tui in tuis}
        return sorted(groups)
        
    # === Graph Construction ===
    def build_mesh_graph_expanded(self):
        """
        Build an expanded MeSH graph using PAR, RB, and RN relationships.
        Returns a directed NetworkX graph.
        """
        G = nx.DiGraph()

        if not hasattr(self, 'cursor') or self.cursor is None:
            raise ValueError("⚠️ No database cursor available. Are you connected?")

        try:
            query = """
            SELECT CUI1, CUI2, REL
            FROM MRREL
            WHERE SAB = 'MSH' AND REL IN ('PAR', 'RB', 'RN')
            """
            self.cursor.execute(query)
            relations = self.cursor.fetchall()

            if not relations:
                print("⚠️ No relevant relations found for expansion.")
                return G

            for cui1, cui2, rel in relations:
                if rel in {"PAR", "RB"}:
                    G.add_edge(cui2, cui1, rel=rel)  # Parent ➔ Child
                elif rel == "RN":
                    G.add_edge(cui1, cui2, rel=rel)  # Narrower ➔ Broader

            print(f"✅ Expanded MeSH graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            return G

        except mysql.connector.Error as err:
            raise RuntimeError(f"❌ MySQL error during expanded graph build: {err}")
        except Exception as e:
            raise RuntimeError(f"❌ Unexpected error during expanded graph build: {e}")

    def build_improved_snomedct_graph(self, limit=None):
        """
        Build a directed graph of SNOMED CT concepts that better captures the ontology structure.

        Parameters:
        - limit: Optional limit on number of MRHIER/MRREL records to process (default: None = no limit)

        Returns:
        - G: NetworkX DiGraph representing the SNOMED CT hierarchy
        - roots: List of root CUIs
        """
        print("🔧 Building comprehensive SNOMED CT graph...")
        
        if not hasattr(self, 'cursor') or self.cursor is None:
            raise ValueError("⚠️ No database cursor available. Are you connected?")

        G = nx.DiGraph()

        try:
            # --- Step 1: Find root CUIs
            print("📊 Identifying SNOMED CT root concepts...")
            root_query = """
                SELECT DISTINCT CUI
                FROM MRHIER
                WHERE SAB = 'SNOMEDCT_US' AND PAUI IS NULL
            """
            self.cursor.execute(root_query)
            root_results = self.cursor.fetchall()
            roots = [r[0] for r in root_results if r[0]]
            print(f"✅ Found {len(roots)} potential root concepts")

            if not roots:
                print("⚠️ No roots via PAUI. Trying CODE 138875005...")
                self.cursor.execute("""
                    SELECT DISTINCT CUI FROM MRCONSO
                    WHERE SAB = 'SNOMEDCT_US' AND CODE = '138875005'
                """)
                root_results = self.cursor.fetchall()
                roots = [r[0] for r in root_results if r[0]]
                print(f"✅ Found {len(roots)} roots via CODE 138875005")

            for root in roots:
                G.add_node(root)

            # --- Step 2: Build edges from MRHIER
            print("📊 Building hierarchy from MRHIER...")
            count_query = "SELECT COUNT(*) FROM MRHIER WHERE SAB = 'SNOMEDCT_US'"
            if limit: count_query += f" LIMIT {limit}"
            self.cursor.execute(count_query)
            total_records = self.cursor.fetchone()[0]

            hier_query = "SELECT CUI, PAUI, AUI, PTR FROM MRHIER WHERE SAB = 'SNOMEDCT_US'"
            if limit: hier_query += f" LIMIT {limit}"
            self.cursor.execute(hier_query)

            aui_to_cui = {}
            batch_size = 10000

            with tqdm(total=total_records, desc="Processing MRHIER") as pbar:
                while True:
                    batch = self.cursor.fetchmany(batch_size)
                    if not batch:
                        break
                    for cui, paui, aui, ptr in batch:
                        if not cui or not aui:
                            continue
                        aui_to_cui[aui] = cui
                        if ptr:
                            path_auis = ptr.split('.')
                            for i in range(len(path_auis) - 1):
                                parent = aui_to_cui.get(path_auis[i])
                                child = aui_to_cui.get(path_auis[i + 1])
                                if parent and child:
                                    G.add_edge(parent, child)
                    pbar.update(len(batch))

            # --- Step 3: Add IS-A relationships
            print("📊 Adding IS-A relationships from MRREL...")
            self.cursor.execute("SELECT COUNT(*) FROM MRREL WHERE SAB = 'SNOMEDCT_US' AND RELA = 'is_a'")
            isa_total = self.cursor.fetchone()[0]

            self.cursor.execute("""
                SELECT CUI1, CUI2
                FROM MRREL
                WHERE SAB = 'SNOMEDCT_US' AND RELA = 'is_a'
            """)
            
            with tqdm(total=isa_total, desc="Adding IS-A") as pbar:
                while True:
                    batch = self.cursor.fetchmany(batch_size)
                    if not batch:
                        break
                    for child, parent in batch:
                        if child and parent:
                            G.add_edge(parent, child)
                    pbar.update(len(batch))

            # --- Step 4: Add PAR/CHD relationships
            print("📊 Adding PAR/CHD relationships...")
            self.cursor.execute("SELECT COUNT(*) FROM MRREL WHERE SAB = 'SNOMEDCT_US' AND REL IN ('PAR', 'CHD')")
            rel_total = self.cursor.fetchone()[0]

            self.cursor.execute("""
                SELECT CUI1, CUI2, REL
                FROM MRREL
                WHERE SAB = 'SNOMEDCT_US' AND REL IN ('PAR', 'CHD')
            """)
            
            with tqdm(total=rel_total, desc="Adding PAR/CHD") as pbar:
                while True:
                    batch = self.cursor.fetchmany(batch_size)
                    if not batch:
                        break
                    for cui1, cui2, rel in batch:
                        if rel == 'PAR' and cui2 and cui1:
                            G.add_edge(cui2, cui1)
                        elif rel == 'CHD' and cui1 and cui2:
                            G.add_edge(cui1, cui2)
                    pbar.update(len(batch))

            # --- Step 5: Add selected attribute relationships
            print("📊 Adding attribute relationships...")
            self.cursor.execute("""
                SELECT COUNT(*)
                FROM MRREL
                WHERE SAB = 'SNOMEDCT_US' AND RELA IN ('finding_site', 'has_active_ingredient', 'causative_agent', 'associated_with')
            """)
            attr_total = self.cursor.fetchone()[0]

            self.cursor.execute("""
                SELECT CUI1, CUI2
                FROM MRREL
                WHERE SAB = 'SNOMEDCT_US' 
                AND RELA IN ('finding_site', 'has_active_ingredient', 'causative_agent', 'associated_with')
            """)

            with tqdm(total=attr_total, desc="Adding attributes") as pbar:
                while True:
                    batch = self.cursor.fetchmany(batch_size)
                    if not batch:
                        break
                    for cui1, cui2 in batch:
                        if cui1 and cui2:
                            G.add_edge(cui1, cui2, rel_type='attribute')
                            G.add_edge(cui2, cui1, rel_type='attribute')
                    pbar.update(len(batch))

            # --- Step 6: Final root identification
            print("📊 Identifying root nodes...")
            non_attr_incoming = lambda n: sum(1 for _, _, d in G.in_edges(n, data=True) if d.get('rel_type') != 'attribute')

            updated_roots = []
            with tqdm(total=G.number_of_nodes(), desc="Finding roots") as pbar:
                for node in G.nodes():
                    if non_attr_incoming(node) == 0:
                        updated_roots.append(node)
                    pbar.update(1)

            print(f"✅ Final SNOMED CT graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            print(f"✅ Found {len(updated_roots)} root nodes")

            if not updated_roots and roots:
                updated_roots = roots
            
            return G, updated_roots

        except mysql.connector.Error as err:
            print(f"❌ MySQL error during SNOMED CT graph building: {err}")
            return G, roots if 'roots' in locals() else []
        except Exception as e:
            print(f"❌ Unexpected error during SNOMED CT graph building: {e}")
            traceback.print_exc()
            return G, roots if 'roots' in locals() else []

    def save_neo4j_subgraph_as_html(    
            self,
            uri,
            user,
            password,
            output_path="neo4j_knowledge_graph.html",
            limit=50
        ):
            """
            Connects to a Neo4j instance and saves a subgraph visualization as HTML using PyVis.

            Parameters:
            - uri (str): Bolt+TLS URI for Neo4j Aura (e.g., neo4j+s://your-instance.databases.neo4j.io)
            - user (str): Neo4j username
            - password (str): Neo4j password
            - output_path (str): Path to save the HTML file
            - limit (int): Max number of edges to visualize
            """
            driver = GraphDatabase.driver(uri, auth=(user, password))
            net = Network(height="700px", width="100%", directed=True, cdn_resources="in_line")

            with driver.session() as session:
                result = session.run(f"""
                    MATCH (n:Concept)-[r]->(m:Concept)
                    RETURN n, r, m
                    LIMIT {limit}
                """)
                for record in result:
                    n = record["n"]
                    m = record["m"]
                    r = record["r"]

                    # Add nodes with labels and tooltips
                    net.add_node(n["cui"], label=n.get("str", n["cui"]), title=n.get("def", ""))
                    net.add_node(m["cui"], label=m.get("str", m["cui"]), title=m.get("def", ""))
                    net.add_edge(n["cui"], m["cui"], label=r.type)

            net.save_graph(output_path)
            print(f"✅ Neo4j subgraph visualization saved to: {output_path}")

    def build_hierarchical_mesh_graph(self):
        """
        Build a hierarchical graph for MeSH using tree numbers.
        Captures true parent-child structure based on the MRSAT.MN field.
        """
        if not hasattr(self, 'cursor') or self.cursor is None:
            raise ValueError("⚠️ No database cursor available. Are you connected?")
        
        print("🔧 Building MeSH hierarchy from tree numbers...")
        
        query = """
        SELECT CUI, ATV AS tree_num FROM MRSAT
        WHERE SAB = 'MSH' AND ATN = 'MN'
        """

        try:
            self.cursor.execute(query)
            results = self.cursor.fetchall()
            
            if not results:
                print("⚠️ No results found in MRSAT for MeSH hierarchy.")
                return None, []
        except mysql.connector.Error as err:
            raise RuntimeError(f"⚠️ MySQL error during hierarchy query: {err}")
        except Exception as e:
            raise RuntimeError(f"⚠️ Unexpected error during hierarchy query: {e}")

        # Build the tree mappings
        tree_to_cui = {}
        cui_to_trees = defaultdict(list)
        for cui, tree_num in results:
            if tree_num and cui:
                tree_to_cui[tree_num] = cui
                cui_to_trees[cui].append(tree_num)
        
        # Build the graph
        G = nx.DiGraph()
        for tree_num, cui in tree_to_cui.items():
            G.add_node(cui)
            parts = tree_num.split('.')
            if len(parts) > 1:
                parent_tree = '.'.join(parts[:-1])
                parent_cui = tree_to_cui.get(parent_tree)
                if parent_cui:
                    G.add_edge(parent_cui, cui)
        
        if G.number_of_nodes() == 0:
            print("⚠️ Warning: Built graph has no nodes!")
        
        print(f"✅ MeSH graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        roots = [node for node in G.nodes() if G.in_degree(node) == 0]
        print(f"✅ Found {len(roots)} root nodes")
        
        return G, roots

    def build_mesh_graph_from_mrrel(self):
        """
        Build a directed graph of MeSH concepts using parent-child relationships from MRREL.
        Only includes 'PAR' (parent) relationships within MeSH (SAB='MSH').
        """
        if not hasattr(self, 'cursor') or self.cursor is None:
            raise ValueError("⚠️ No database cursor available. Are you connected?")

        print("🔧 Building MeSH graph from MRREL...")
        G = nx.DiGraph()

        query = """
        SELECT CUI1 AS child, CUI2 AS parent
        FROM MRREL
        WHERE SAB = 'MSH' AND REL = 'PAR'
        """

        try:
            self.cursor.execute(query)
            results = self.cursor.fetchall()
            
            if not results:
                print("⚠️ No parent-child relationships found in MRREL for MSH.")
                return G  # Return empty graph

            for child, parent in results:
                if child and parent:
                    G.add_edge(parent, child)

        except mysql.connector.Error as err:
            raise RuntimeError(f"❌ MySQL error while building MeSH graph from MRREL: {err}")
        except Exception as e:
            raise RuntimeError(f"❌ Unexpected error while building MeSH graph: {e}")

        print(f"✅ MeSH graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    # === NER & Concept Extraction ===
    def load_ner_model(self, model_name='dslim/bert-base-NER'):
        """
        Load a HuggingFace NER model for extracting entities from user input.
        """
        self.ner_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.ner_model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.ner_pipeline = pipeline("ner", model=self.ner_model, tokenizer=self.ner_tokenizer, aggregation_strategy="average")

    def extract_entities_from_text(self, text):
        """
        Run NER on the input text and extract recognized entities.
        Returns a list of unique terms.
        """
        if not hasattr(self, 'ner_pipeline'):
            self.load_ner_model()  # default model if not loaded
        entities = self.ner_pipeline(text)
        terms = list({ent['word'] for ent in entities if ent['score'] > 0.5})
        return terms

    def generate_concept_pairs_from_text(self, text, sab="MSH"):
        """
        Given free-form text, run NER and convert entities into concept pairs using CUIs.
        Returns a list of (term1, term2) pairs.
        """
        terms = self.extract_entities_from_text(text)
        print("🔍 Extracted Entities:", terms)  # Print entities for inspection
        if len(terms) < 2:
            return []
        pairs = [(terms[i], terms[j]) for i in range(len(terms)) for j in range(i+1, len(terms))]
        return pairs
    
    def run_pipeline_on_text(self, text, UG, tui_to_group, sab="MSH", max_depth=20, verbose=False, similarity_metrics=["path", "wup", "lch"]):
        """
        Full pipeline to go from free text -> concept pairs -> semantic similarity analysis DataFrame.
        """
        concept_pairs = self.generate_concept_pairs_from_text(text, sab=sab)
        if not concept_pairs:
            print("No valid concept pairs extracted from text.")
            return pd.DataFrame()
        return self.generate_concept_path_analysis(concept_pairs, UG, tui_to_group, sab=sab, max_depth=max_depth, verbose=False, similarity_metrics=similarity_metrics)

    # === Word Frequency and IC Pipeline ===
    def compute_word_frequencies_to_parquet(
        self,
        parquet_output="word_counts_pubmed.parquet",
        dataset_name="yanjx21/PubMed",
        num_abstracts=3723949,
        max_ngram=3,
        batch_size=100_000
        ):
            stop_words = set(stopwords.words("english"))
            dataset = load_dataset(dataset_name, split="train", streaming=True)
            word_counter = Counter()
            total_words = 0
            pbar = tqdm(total=num_abstracts, desc="📝 Processing abstracts", dynamic_ncols=True)

            writer = None
            batch = []

            for i, example in enumerate(dataset):
                if i >= num_abstracts:
                    break

                text = example.get("text", "").lower()
                tokens = re.findall(r"\b\w[\w-]*\b", text)
                tokens = [t for t in tokens if t not in stop_words]
                total_words += len(tokens)

                for n in range(1, max_ngram + 1):
                    for ng in ngrams(tokens, n):
                        word_counter[" ".join(ng)] += 1

                if (i + 1) % batch_size == 0:
                    batch_df = pd.DataFrame(word_counter.items(), columns=["Word", "Count"])
                    table = pa.Table.from_pandas(batch_df)
                    if writer is None:
                        writer = pq.ParquetWriter(parquet_output, table.schema)
                    writer.write_table(table)
                    word_counter.clear()  # reset counter for next batch

                pbar.update(1)

            # Final batch
            if word_counter:
                batch_df = pd.DataFrame(word_counter.items(), columns=["Word", "Count"])
                table = pa.Table.from_pandas(batch_df)
                if writer is None:
                    writer = pq.ParquetWriter(parquet_output, table.schema)
                writer.write_table(table)

            if writer:
                writer.close()

            pbar.close()
            print(f"✅ Completed processing {num_abstracts:,} abstracts")
            print(f"💾 Word frequencies saved to {parquet_output}")

    def map_words_to_cuis(self, word_counts_parquet="word_counts_pubmed.parquet", output_path="cui_counts_pubmed.parquet", sab=["MSH", "SNOMEDCT_US"], batch_size=100000, max_entries=None):
        dataset = ds.dataset(word_counts_parquet, format="parquet")
        scanner = dataset.scanner(columns=["Word", "Count"], batch_size=batch_size)
        cui_counter = Counter()
        processed = 0
        pbar = tqdm(desc="🔗 Mapping words to CUIs", total=max_entries)

        for batch in scanner.to_batches():
            df = batch.to_pandas()
            for word, count in zip(df["Word"], df["Count"]):
                if max_entries and processed >= max_entries:
                    break
                cui = self.find_cui(word, sab=sab)
                if cui:
                    cui_counter[cui] += count
                processed += 1
                pbar.update(1)
            if max_entries and processed >= max_entries:
                break

        pbar.close()
        df_cui = pd.DataFrame(cui_counter.items(), columns=["CUI", "Count"])
        table = pa.Table.from_pandas(df_cui)
        pq.write_table(table, output_path, compression="snappy")
        print(f"✅ Saved CUI counts to {output_path}")
        return df_cui

    def load_cui_ic_from_parquet(self, parquet_path="cui_ic_pubmed.parquet"):
        """
        Load CUI → Information Content (IC) values from a Parquet file and store in self.cui_to_ic.
        """
        try:
            df = pd.read_parquet(parquet_path)
            self.cui_to_ic = dict(zip(df["CUI"], df["IC"]))
            print(f"✅ Loaded {len(self.cui_to_ic):,} CUI IC values from {parquet_path}")
        except Exception as e:
            print(f"❌ Failed to load IC values: {e}")

    def compute_information_content(self, cui_count_df, output_parquet="cui_ic_pubmed.parquet"):
        total = cui_count_df["Count"].sum()
        cui_count_df["IC"] = -np.log(cui_count_df["Count"] / total)
        table = pa.Table.from_pandas(cui_count_df)
        pq.write_table(table, output_parquet, compression="snappy")
        print(f"🧠 IC values saved to {output_parquet}")
        return cui_count_df

    def run_ngram_to_ic_pipeline(self,
                                parquet_output="word_counts_pubmed.parquet",
                                cui_output="cui_counts_pubmed.parquet",
                                ic_output="cui_ic_pubmed.parquet",
                                dataset_name="yanjx21/PubMed",
                                sab=["MSH", "SNOMEDCT_US"],
                                max_entries=None,
                                num_abstracts=3723949,
                                max_ngram=3,
                                batch_size=100_000):
        # Step 1: Extract and stream word counts directly to Parquet
        self.compute_word_frequencies_to_parquet(
            parquet_output=parquet_output,
            dataset_name=dataset_name,
            num_abstracts=num_abstracts,
            max_ngram=max_ngram,
            batch_size=batch_size
        )

        # Step 2: Map to CUIs
        cui_df = self.map_words_to_cuis(
            word_counts_parquet=parquet_output,
            output_path=cui_output,
            sab=sab,
            max_entries=max_entries
        )

        # Step 3: Compute IC
        ic_df = self.compute_information_content(
            cui_df,
            output_parquet=ic_output
        )

        # Store for similarity metrics
        self.cui_to_ic = dict(zip(ic_df["CUI"], ic_df["IC"]))
        return ic_df

    # === Semantic Similarity Metrics ===
    def semantic_similarity_path(self, cui1, cui2, UG):
        """
        Computes semantic similarity between two CUIs based on shortest path length.
        
        Parameters:
        - cui1, cui2: CUIs
        - UG: Undirected NetworkX graph (ontology graph)
        
        Returns:
        - similarity score (float between 0 and 1)
        - path length (integer)
        """
        try:
            path_length = nx.shortest_path_length(UG, source=cui1, target=cui2)
            similarity = 1 / (1 + path_length)
            return similarity, path_length
        except nx.NetworkXNoPath:
            return 0.0, None  # No path found, similarity = 0
        except Exception as e:
            print(f"⚠️ Error computing path similarity between {cui1} and {cui2}: {e}")
            return 0.0, None

    def semantic_similarity_wu_palmer(self, cui1, cui2, UG):
        """
        Compute Wu-Palmer similarity between two CUIs based on
        their Lowest Common Subsumer (LCS) depth.

        Parameters:
        - cui1, cui2: CUIs (concept identifiers)
        - UG: Undirected graph representing the ontology

        Returns:
        - Wu-Palmer similarity (float between 0 and 1)
        """
        try:
            # Step 1: Identify a root node
            root_nodes = [node for node, degree in UG.degree() if degree == 1]
            if not root_nodes:
                print("⚠️ No root nodes found in the ontology graph.")
                return 0.0
            root = root_nodes[0]

            # Step 2: Find paths from root to each concept
            path1 = nx.shortest_path(UG, source=root, target=cui1)
            path2 = nx.shortest_path(UG, source=root, target=cui2)

            # Step 3: Identify common ancestors
            common_ancestors = set(path1).intersection(path2)
            if not common_ancestors:
                print(f"⚠️ No common ancestor between '{cui1}' and '{cui2}'.")
                return 0.0

            # Step 4: Pick the deepest common ancestor (maximum depth)
            lcs = None
            max_depth = -1
            for node in common_ancestors:
                try:
                    depth = nx.shortest_path_length(UG, source=root, target=node)
                    if depth > max_depth:
                        max_depth = depth
                        lcs = node
                except Exception:
                    continue

            if lcs is None:
                print(f"⚠️ Could not determine a valid LCS between '{cui1}' and '{cui2}'.")
                return 0.0

            # Step 5: Calculate depths
            depth1 = nx.shortest_path_length(UG, source=root, target=cui1)
            depth2 = nx.shortest_path_length(UG, source=root, target=cui2)

            # Step 6: Wu-Palmer similarity formula
            similarity = (2 * max_depth) / (depth1 + depth2)
            return similarity

        except nx.NetworkXNoPath:
            print(f"⚠️ No path found between '{cui1}' or '{cui2}' and the root.")
            return 0.0
        except Exception as e:
            print(f"⚠️ Error computing Wu-Palmer similarity between {cui1} and {cui2}: {e}")
            return 0.0

    def semantic_similarity_lch(self, cui1, cui2, UG, max_depth=20):
        """
        Compute Leacock-Chodorow similarity between two CUIs.
        
        Parameters:
        - cui1, cui2: CUIs
        - UG: Undirected graph
        - max_depth: Estimated maximum depth of the ontology (default 20)
        
        Returns:
        - similarity (float)
        """
        try:
            path_length = nx.shortest_path_length(UG, source=cui1, target=cui2)
            similarity = -math.log((path_length) / (2.0 * max_depth))
            return similarity
        except Exception as e:
            print(f"⚠️ Error computing LCH similarity between {cui1} and {cui2}: {e}")
            return 0.0

    def semantic_similarity_lin(self, cui1, cui2, lcs_cui):
        """
        Compute Lin similarity between two CUIs using a known LCS CUI and IC values.

        Parameters:
        - cui1: First CUI
        - cui2: Second CUI
        - lcs_cui: Precomputed Lowest Common Subsumer (CUI)

        Returns:
        - float: Lin similarity (0 to 1) or 0.0 if IC is missing
        """
        try:
            if not hasattr(self, "cui_to_ic") or not self.cui_to_ic:
                print("⚠️ IC values not loaded")
                return 0.0

            ic_lcs = self.cui_to_ic.get(lcs_cui)
            ic1 = self.cui_to_ic.get(cui1)
            ic2 = self.cui_to_ic.get(cui2)

            if ic_lcs is None or ic1 is None or ic2 is None:
                return 0.0

            return (2 * ic_lcs) / (ic1 + ic2)
        except Exception as e:
            print(f"⚠️ Lin similarity error for {cui1}, {cui2}, LCS={lcs_cui}: {e}")
            return 0.0

    def semantic_similarity_jiang_conrath(self, cui1, cui2, UG, cui_to_ic):
        """
        Compute Jiang-Conrath distance between two CUIs.

        Returns:
        - float: JC distance (lower = more similar; may be negative if IC(LCS) > IC(concepts))
        """
        try:
            lcs_cui, _ = self.find_lcs_from_shortest_path(cui1, cui2, UG)
            if not lcs_cui:
                return None  # signify missing path instead of 0.0

            ic1 = cui_to_ic.get(cui1)
            ic2 = cui_to_ic.get(cui2)
            ic_lcs = cui_to_ic.get(lcs_cui)

            if ic1 is None or ic2 is None or ic_lcs is None:
                return None

            return ic1 + ic2 - 2 * ic_lcs
        except Exception as e:
            print(f"⚠️ Jiang-Conrath distance error for {cui1}, {cui2}: {e}")
            return None

    def semantic_similarity_resnik(self, cui1, cui2, UG):
        """
        Compute Resnik similarity using LCS IC value from self.cui_to_ic.
        """
        if not hasattr(self, "cui_to_ic") or not self.cui_to_ic:
            print("⚠️ IC values not loaded. Call `load_cui_ic_from_parquet()` first.")
            return 0.0

        lcs_cui, _ = self.find_lcs_from_shortest_path(cui1, cui2, UG)
        if not lcs_cui:
            print(f"⚠️ No LCS found between {cui1} and {cui2}")
            return 0.0

        return self.cui_to_ic.get(lcs_cui, 0.0)

    def cosine_similarity(self, vec1, vec2):
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
  
    def normalize_similarity_scores(self, df, max_depth=20):
        """
        Normalize similarity scores to [0, 1] range.

        Parameters:
        - df (pd.DataFrame): DataFrame with similarity columns
        - max_depth (int): Used for LCH normalization
        
        Returns:
        - pd.DataFrame with _NORM columns for each metric
        """
        df = df.copy()

        # Precompute max values
        max_lch = -math.log(1 / (2 * max_depth))  # LCH theoretical max
        max_ic = max(self.cui_to_ic.values()) if hasattr(self, "cui_to_ic") and self.cui_to_ic else 1.0

        possible_metrics = [
            'SIM_PATH', 'SIM_WUP', 'SIM_LCH',
            'SIM_RESNIK', 'SIM_LIN', 'SIM_JCN',
            'SIM_W2V', 'SIM_GLOVE', 'SIM_TRANSFORMER'
        ]

        for metric in possible_metrics:
            if metric not in df.columns:
                continue

            if metric in ['SIM_PATH', 'SIM_WUP']:
                df[f"{metric}_NORM"] = df[metric]

            elif metric == 'SIM_LCH':
                df[f"{metric}_NORM"] = df[metric] / max_lch
                #df[f"{metric}_NORM"] = df[f"{metric}_NORM"].clip(0, 1)

            elif metric == 'SIM_RESNIK':
                df[f"{metric}_NORM"] = df[metric] / max_ic
                #df[f"{metric}_NORM"] = df[f"{metric}_NORM"].clip(0, 1)

            elif metric == 'SIM_LIN':
                df[f"{metric}_NORM"] = df[metric] / max_ic

            elif metric == 'SIM_JCN':
                # Use 1 / (1 + d) form, then normalize to [0, 1]
                def squash_jcn(val):
                    if val is None or not isinstance(val, (int, float)):
                        return np.nan
                    return 1 / (1 + max(val, 0))
                df[f"{metric}_NORM"] = df[metric].apply(squash_jcn)

            elif metric in ['SIM_W2V', 'SIM_GLOVE', 'SIM_TRANSFORMER']:
                valid_entries = df[metric].apply(lambda x: isinstance(x, (float, int, np.float32, np.float64)))
                df.loc[valid_entries, f"{metric}_NORM"] = (df.loc[valid_entries, metric] + 1) / 2
                df[f"{metric}_NORM"] = df[f"{metric}_NORM"].clip(0, 1)

        return df

    # === Embedding-Based Similarity ===
    def embed_text_word2vec(self, text):
        if text == "No definition found.":
            return np.nan
        words = text.split()
        vectors = [self.word2vec_model[word] for word in words if word in self.word2vec_model]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.word2vec_model.vector_size)

    def embed_text_glove(self, text):
        if text == "No definition found.":
            return np.nan
        words = text.split()
        vectors = [self.glove_model[word] for word in words if word in self.glove_model]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.glove_model.vector_size)

    def embed_text_transformer(self, text):
        if text == "No definition found.":
            return np.nan
        return self.transformer_model.encode(text)

    # === Path & Similarity Analysis ===
    def generate_concept_path_analysis(self, concept_pairs, UG, tui_to_group, sab="MSH", max_depth=20, similarity_metrics=None, UG_expanded=None, verbose=True):
        """
        Given a list of concept pairs (CUIs or terms), computes paths, LCS, STRs, DEFs, 
        semantic groups, and requested semantic similarity metrics.

        Parameters:
        - concept_pairs: list of (cui_or_term1, cui_or_term2) tuples
        - UG: undirected NetworkX graph of the ontology
        - tui_to_group: dict mapping TUI to semantic group
        - sab: source vocabulary (default: "MSH")
        - max_depth: estimated max depth for LCH computation
        - similarity_metrics: list of metric names to calculate ("path", "wup", "lch", "w2v", "glove", "transformer")
        - verbose: whether to print warnings/info (default: True)

        Returns:
        - pandas.DataFrame
        """

        def vprint(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)

        if similarity_metrics is None:
            similarity_metrics = ["path", "wup", "lch", "resnik", "lin", "transformer"] #"jiang_conrath"

        results = []

        for input1, input2 in tqdm(concept_pairs, desc="Analyzing concept pairs"):
            try:
                cui1 = self.resolve_cui(input1, sab=sab)
                cui2 = self.resolve_cui(input2, sab=sab)

                if not cui1 or not cui2:
                    vprint(f"⚠️ Could not resolve CUI for '{input1}' or '{input2}'")
                    continue

                str1 = self.get_str(cui1, sab)
                str2 = self.get_str(cui2, sab)
                def1 = self.get_def(cui1, sab)
                def2 = self.get_def(cui2, sab)

                # Try shortest path first in UG, then fallback to UG_expanded if provided
                try:
                    path = nx.shortest_path(UG, source=cui1, target=cui2)
                    graph_used = UG
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    if UG_expanded is not None:
                        try:
                            path = nx.shortest_path(UG_expanded, source=cui1, target=cui2)
                            graph_used = UG_expanded
                            vprint(f"ℹ️ Path found using expanded graph between '{input1}' and '{input2}'")
                        except (nx.NetworkXNoPath, nx.NodeNotFound):
                            vprint(f"⚠️ No path found between '{input1}' and '{input2}' even in expanded graph.")
                            path = None
                            graph_used = None
                    else:
                        vprint(f"⚠️ No path found between '{input1}' and '{input2}' (no expanded graph provided).")
                        path = None
                        graph_used = None

                path_length = len(path) - 1
                named_path = [(cui, self.get_str(cui, sab)) for cui in path]
                if graph_used is not None:
                    lcs_cui, lcs_depth = self.find_lcs_from_shortest_path(cui1, cui2, graph_used)
                else:
                    lcs_cui, lcs_depth = None, None
                lcs_str = self.get_str(lcs_cui, sab) if lcs_cui else None

                source_groups = self.get_semantic_groups_by_cui(cui1, tui_to_group)
                target_groups = self.get_semantic_groups_by_cui(cui2, tui_to_group)
                lcs_groups = self.get_semantic_groups_by_cui(lcs_cui, tui_to_group) if lcs_cui else ["Group not found"]

                result_row = {
                    'SOURCE_INPUT': input1,
                    'SOURCE_CUI': cui1,
                    'SOURCE_STR': str1,
                    'SOURCE_DEF': def1,
                    'SOURCE_GROUPS': '; '.join(source_groups),
                    'TARGET_INPUT': input2,
                    'TARGET_CUI': cui2,
                    'TARGET_STR': str2,
                    'TARGET_DEF': def2,
                    'TARGET_GROUPS': '; '.join(target_groups),
                    'LCS_CUI': lcs_cui,
                    'LCS_STR': lcs_str,
                    'LCS_GROUPS': '; '.join(lcs_groups),
                    'LCS_DEPTH': lcs_depth,
                    'PATH_LENGTH': path_length,
                    'PATH_SEQUENCE': ' → '.join([f"{cui} ({name})" for cui, name in named_path]),
                }

                # Add only requested metrics (use the correct graph!)
                if "path" in similarity_metrics and graph_used:
                    sim_path, _ = self.semantic_similarity_path(cui1, cui2, graph_used)
                    result_row['SIM_PATH'] = sim_path

                if "wup" in similarity_metrics and graph_used:
                    result_row['SIM_WUP'] = self.semantic_similarity_wu_palmer(cui1, cui2, graph_used)

                if "lch" in similarity_metrics and graph_used:
                    result_row['SIM_LCH'] = self.semantic_similarity_lch(cui1, cui2, graph_used, max_depth=max_depth)

                if "w2v" in similarity_metrics:
                    if hasattr(self, "word2vec_model"):
                        vec1 = self.embed_text_word2vec(def1)
                        vec2 = self.embed_text_word2vec(def2)
                        result_row['SIM_W2V'] = self.cosine_similarity(vec1, vec2)
                    else:
                        print(f"⚠️ Word2Vec model not loaded — skipping SIM_W2V for ({input1}, {input2})")

                if "glove" in similarity_metrics:
                    if hasattr(self, "glove_model"):
                        vec1 = self.embed_text_glove(def1)
                        vec2 = self.embed_text_glove(def2)
                        result_row['SIM_GLOVE'] = self.cosine_similarity(vec1, vec2)
                    else:
                        print(f"⚠️ GloVe model not loaded — skipping SIM_GLOVE for ({input1}, {input2})")

                if "transformer" in similarity_metrics:
                    if hasattr(self, "transformer_model"):
                        vec1 = self.embed_text_transformer(def1)
                        vec2 = self.embed_text_transformer(def2)
                        result_row['SIM_TRANSFORMER'] = self.cosine_similarity(vec1, vec2)
                    else:
                        print(f"⚠️ Transformer model not loaded — skipping SIM_TRANSFORMER for ({input1}, {input2})")
                
                if "resnik" in similarity_metrics and graph_used:
                    if hasattr(self, "cui_to_ic") and self.cui_to_ic:
                        try:
                            result_row['SIM_RESNIK'] = self.semantic_similarity_resnik(cui1, cui2, graph_used)
                        except Exception as e:
                            result_row['SIM_RESNIK'] = None
                            vprint(f"⚠️ Resnik similarity error for ({input1}, {input2}): {e}")

                # Lin Similarity
                if "lin" in similarity_metrics and graph_used:
                    if hasattr(self, "cui_to_ic") and self.cui_to_ic:
                        try:
                            result_row['SIM_LIN'] = self.semantic_similarity_lin(cui1, cui2, lcs_cui)
                        except Exception as e:
                            result_row['SIM_LIN'] = None
                            vprint(f"⚠️ Lin similarity error for ({input1}, {input2}): {e}")

                # Jiang-Conrath Similarity
                if "jiang_conrath" in similarity_metrics and graph_used:
                    if hasattr(self, "cui_to_ic") and self.cui_to_ic:
                        result_row['SIM_JCN'] = self.semantic_similarity_jiang_conrath(cui1, cui2, graph_used)
                    else:
                        print(f"⚠️ IC values not loaded — skipping SIM_JCN for ({input1}, {input2})")

                results.append(result_row)

            except nx.NetworkXNoPath:
                vprint(f"⚠️ No path found between '{input1}' and '{input2}'")
                results.append({
                    'SOURCE_INPUT': input1,
                    'SOURCE_CUI': cui1 if 'cui1' in locals() else None,
                    'SOURCE_STR': str1 if 'str1' in locals() else None,
                    'SOURCE_DEF': def1 if 'def1' in locals() else None,
                    'SOURCE_GROUPS': None,
                    'TARGET_INPUT': input2,
                    'TARGET_CUI': cui2 if 'cui2' in locals() else None,
                    'TARGET_STR': str2 if 'str2' in locals() else None,
                    'TARGET_DEF': def2 if 'def2' in locals() else None,
                    'TARGET_GROUPS': None,
                    'LCS_CUI': None,
                    'LCS_STR': None,
                    'LCS_GROUPS': None,
                    'LCS_DEPTH': None,
                    'PATH_LENGTH': None,
                    'PATH_SEQUENCE': "No path found",
                })
            except Exception as e:
                vprint(f"⚠️ Unexpected error for pair ({input1}, {input2}): {e}")
                results.append({
                    'SOURCE_INPUT': input1,
                    'SOURCE_CUI': cui1 if 'cui1' in locals() else None,
                    'SOURCE_STR': str1 if 'str1' in locals() else None,
                    'SOURCE_DEF': def1 if 'def1' in locals() else None,
                    'SOURCE_GROUPS': None,
                    'TARGET_INPUT': input2,
                    'TARGET_CUI': cui2 if 'cui2' in locals() else None,
                    'TARGET_STR': str2 if 'str2' in locals() else None,
                    'TARGET_DEF': def2 if 'def2' in locals() else None,
                    'TARGET_GROUPS': None,
                    'LCS_CUI': None,
                    'LCS_STR': None,
                    'LCS_GROUPS': None,
                    'LCS_DEPTH': None,
                    'PATH_LENGTH': None,
                    'PATH_SEQUENCE': f"Error: {str(e)}",
                })

        return pd.DataFrame(results)

    def find_lcs_from_shortest_path(self, cui1, cui2, UG):
        """
        Find the Lowest Common Subsumer (LCS) between two CUIs as the
        deepest node on their shortest path, excluding the terminals.

        Parameters:
        - cui1, cui2: CUIs of the two concepts
        - UG: Undirected graph representing the ontology

        Returns:
        - lcs_cui: The CUI of the lowest common subsumer
        - lcs_depth: Its depth from the root
        """
        try:
            # 1) Identify a root (degree == 1 heuristic)
            roots = [n for n, d in UG.degree() if d == 1]
            if not roots:
                print("⚠️ No root nodes found.")
                return None, None
            root = roots[0]

            # 2) Get the shortest path between the two CUIs
            path = nx.shortest_path(UG, source=cui1, target=cui2)

            # 3) Compute depth of each node on the path
            depths = {}
            for node in path:
                try:
                    depths[node] = nx.shortest_path_length(UG, source=root, target=node)
                except nx.NetworkXNoPath:
                    depths[node] = -1

            # 4) Exclude terminal CUIs unless path is direct
            internal_nodes = path
            if len(path) > 2:
                internal_nodes = path[1:-1]  # remove terminals

            if not internal_nodes:
                return None, None

            # 5) Choose the internal node with max depth
            lcs_cui = max(internal_nodes, key=lambda n: depths[n])
            lcs_depth = depths[lcs_cui]

            return lcs_cui, lcs_depth

        except nx.NetworkXNoPath:
            print(f"⚠️ No path found between '{cui1}' and '{cui2}'")
            return None, None
        except Exception as e:
            print(f"⚠️ Error finding LCS: {e}")
            return None, None
            
    def compute_max_depth(self, UG, roots=None):
        """
        Compute the maximum depth of the ontology by finding the longest shortest path 
        from root nodes to all other nodes.

        Parameters:
        - UG: Undirected or directed NetworkX graph
        - roots: list of root node CUIs (optional). If None, estimate from degree 1 nodes.

        Returns:
        - max_depth (int)
        """
        if roots is None:
            # Assume root nodes are nodes with degree 1
            roots = [node for node, degree in UG.degree() if degree == 1]
        
        max_depth = 0
        for root in roots:
            lengths = nx.single_source_shortest_path_length(UG, root)
            local_max = max(lengths.values())
            if local_max > max_depth:
                max_depth = local_max
        
        return max_depth

    def print_path_between_concepts(self, cui1, cui2, UG):
        """
        Prints the shortest path between two CUIs if one exists in the given graph.

        Parameters:
        - cui1 (str): Starting concept CUI
        - cui2 (str): Target concept CUI
        - UG (networkx.Graph): UMLS ontology graph

        Returns:
        - list of CUIs in the path if found, else None
        """
        if nx.has_path(UG, cui1, cui2):
            path = nx.shortest_path(UG, cui1, cui2)
            print(f"✅ Path from {cui1} to {cui2}:")
            for node in path:
                print(f" - {node} ({self.get_str(node, sab='MSH')})")
            return path
        else:
            print(f"❌ No path found between {cui1} and {cui2}")
            return None

    # === Visualization Tools ===
    def visualize_concept_path_interactive(self, row, save_path="concept_path.html"):
        """
        Create an interactive HTML visualization of a concept path with a built-in legend and tooltips.

        Parameters:
        - row: Row from DataFrame with required columns
        - save_path: Path to save the HTML file
        """
        from pyvis.network import Network
        import tempfile
        import os

        # Extract information
        path_seq = row['PATH_SEQUENCE']
        source_cui = row['SOURCE_CUI']
        target_cui = row['TARGET_CUI']
        lcs_cui = row['LCS_CUI']

        if not path_seq or path_seq == "No path found":
            print("No path available for visualization.")
            return

        # Build the Network
        net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="black", directed=True)
        nodes = []
        edges = []

        parts = path_seq.split(' → ')
        for part in parts:
            match = re.match(r"^(C\d{7}) \((.*)\)$", part.strip())
            if match:
                cui, name = match.groups()
                nodes.append((cui.strip(), name.strip()))
        
        for i in range(len(nodes) - 1):
            edges.append((nodes[i][0], nodes[i+1][0]))

        # Node colors
        color_map = {
            source_cui: 'blue',
            target_cui: 'red',
            lcs_cui: 'green'
        }
        
        for cui, label in nodes:
            str_text = self.get_str(cui, 'MSH')
            def_text = self.get_def(cui, 'MSH')
            groups = self.get_semantic_groups_by_cui(cui, tui_to_group)
            group_text = ', '.join(groups) if groups else "Unknown"

            color = color_map.get(cui, 'lightgray')
            net.add_node(
                cui, 
                label=f"{cui}", 
                title=f"<b>STR:</b> {str_text}<br><b>DEF:</b> {def_text}<br><b>GROUP:</b> {group_text}",
                color=color
            )

        for src, tgt in edges:
            net.add_edge(src, tgt)

        # Temporary HTML save
        temp_html = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
        temp_html.close()
        net.save_graph(temp_html.name)

        # Now read the generated network HTML
        with open(temp_html.name, "r", encoding="utf-8") as f:
            net_html = f.read()

        os.unlink(temp_html.name)  # Clean up temp

        # Build final HTML
        legend_html = """
        <div style="position:absolute; top:10px; right:10px; background:white; border:1px solid gray; padding:10px; border-radius:8px; opacity:0.9;">
            <h4>Legend</h4>
            <p><span style="color:blue;">⬤</span> Source Concept (CUI1)</p>
            <p><span style="color:red;">⬤</span> Target Concept (CUI2)</p>
            <p><span style="color:green;">⬤</span> Lowest Common Subsumer (LCS)</p>
        </div>
        """

        final_html = f"""
        <html>
        <head>
        <meta charset="utf-8">
        <title>Concept Path Visualization</title>
        </head>
        <body>
        {legend_html}
        {net_html}
        </body>
        </html>
        """

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(final_html)

        print(f"✅ Interactive HTML graph saved to {save_path}")
    
    def plot_all_similarity_metrics_heatmap(
            self, 
            df, 
            normalize=True, 
            max_depth=20, 
            figsize_width=10, 
            figsize_height=None
        ):
            """
            Plot a heatmap of semantic similarity scores for all term pairs.

            Parameters:
            - df (pd.DataFrame): The DataFrame containing similarity scores.
            - normalize (bool): Normalize all similarity metrics to [0, 1] range.
            - max_depth (int): Max depth for LCH normalization.
            - figsize_width (int): Width of the figure (default: 10).
            - figsize_height (float or None): Height of the figure. 
            If None, auto-scales based on number of term pairs.
            """
            possible_metrics = [
                'SIM_PATH', 'SIM_WUP', 'SIM_LCH', 'SIM_RESNIK',
                'SIM_LIN', 'SIM_JCN',
                'SIM_W2V', 'SIM_GLOVE', 'SIM_TRANSFORMER'
            ]

            metric_name_mapping = {
                'SIM_PATH': 'Path',
                'SIM_WUP': 'Wu-Palmer',
                'SIM_LCH': 'LCH',
                'SIM_RESNIK': 'Resnik',
                'SIM_LIN': 'Lin',
                'SIM_JCN': 'Jiang-Conrath',
                'SIM_W2V': 'Word2Vec',
                'SIM_GLOVE': 'GloVe',
                'SIM_TRANSFORMER': 'Transformer'
            }

            available_metrics = [m for m in possible_metrics if m in df.columns]
            if not available_metrics:
                print("⚠️ No similarity metrics found in DataFrame to plot.")
                return

            if normalize:
                df = self.normalize_similarity_scores(df, max_depth=max_depth)
                metric_cols = [f"{m}_NORM" for m in available_metrics if f"{m}_NORM" in df.columns]
                label_mapping = {f"{m}_NORM": metric_name_mapping[m] for m in available_metrics if f"{m}_NORM" in df.columns}
            else:
                # Optionally exclude negative-valued JCN if not normalized
                df = df[df["SIM_JCN"] >= 0] if "SIM_JCN" in df.columns else df
                metric_cols = available_metrics
                label_mapping = {m: metric_name_mapping[m] for m in available_metrics}

            if not metric_cols:
                print("⚠️ No available (or normalized) metrics to plot after filtering.")
                return

            df_heatmap = df.copy()
            df_heatmap['Term Pair'] = df_heatmap['SOURCE_STR'] + " ↔ " + df_heatmap['TARGET_STR']
            df_plot = df_heatmap.set_index('Term Pair')[metric_cols]

            df_plot = df_plot.rename(columns=label_mapping)
            df_plot = df_plot.apply(pd.to_numeric, errors='coerce')

            # 🔥 Auto-scale height if not provided
            if figsize_height is None:
                figsize_height = max(8, 0.5 * len(df_plot))

            # Plot
            plt.figure(figsize=(figsize_width, figsize_height))
            sns.heatmap(df_plot, annot=True, cmap="coolwarm", vmin=0, vmax=1 if normalize else None, cbar=True)

            title_suffix = "(Normalized)" if normalize else "(Raw Scores)"
            plt.title(f"Semantic Similarity Heatmap for All Term Pairs {title_suffix}", fontsize=16)
            plt.xlabel("Similarity Metric", fontsize=14)
            plt.ylabel("Term Pair", fontsize=14)
            plt.xticks(rotation=45, ha="right", fontsize=12)
            plt.yticks(fontsize=12)
            plt.tight_layout()
            plt.show()

    def plot_all_similarity_metrics_radar(
        self,
        df: pd.DataFrame,
        normalize: bool = True,
        max_depth: int = 18,
        figsize_per_plot: tuple = (4, 4),
        plots_per_row: int = 3
        ):
            """
            Plots a grid of radar charts for similarity metrics per concept pair.

            Parameters:
            - df (pd.DataFrame): Input DataFrame.
            - normalize (bool): Use normalized metrics.
            - max_depth (int): Max depth used in normalization.
            - figsize_per_plot (tuple): Size of each subplot.
            - plots_per_row (int): Number of radar charts per row.
            """
            if normalize:
                df = self.normalize_similarity_scores(df.copy(), max_depth=max_depth)

            metrics = [
                'SIM_PATH', 'SIM_WUP', 'SIM_LCH', 'SIM_RESNIK',
                'SIM_LIN', 'SIM_JCN', 'SIM_W2V', 'SIM_GLOVE', 'SIM_TRANSFORMER'
            ]

            rows_data = []
            for _, row in df.iterrows():
                labels = []
                values = []
                for metric in metrics:
                    col = f"{metric}_NORM" if normalize and f"{metric}_NORM" in df.columns else metric
                    if col in row and pd.notna(row[col]):
                        labels.append(metric.replace("SIM_", "").title())
                        values.append(float(row[col]))
                if len(labels) >= 3:
                    rows_data.append((row['SOURCE_STR'], row['TARGET_STR'], labels, values))

            num_plots = len(rows_data)
            if num_plots == 0:
                print("No valid pairs with enough metrics to plot.")
                return

            num_cols = plots_per_row
            num_rows = int(np.ceil(num_plots / num_cols))

            fig, axes = plt.subplots(
                num_rows, num_cols,
                figsize=(figsize_per_plot[0]*num_cols, figsize_per_plot[1]*num_rows),
                subplot_kw=dict(polar=True)
            )

            if num_rows == 1 and num_cols == 1:
                axes = np.array([[axes]])
            elif num_rows == 1 or num_cols == 1:
                axes = axes.reshape((num_rows, num_cols))

            for idx, (source, target, labels, values) in enumerate(rows_data):
                row_idx, col_idx = divmod(idx, num_cols)
                ax = axes[row_idx, col_idx]

                angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
                values_closed = values + [values[0]]
                angles_closed = angles + [angles[0]]

                color = self.custom_palette[idx % len(self.custom_palette)]

                ax.plot(angles_closed, values_closed, linewidth=2, color=color)
                ax.fill(angles_closed, values_closed, alpha=0.25, color=color)
                ax.set_xticks(angles)
                ax.set_xticklabels(labels, fontsize=8)
                ax.set_ylim(0, 1)
                ax.set_title(f"{source} ↔ {target}", fontsize=10, pad=10)

            for i in range(num_plots, num_rows * num_cols):
                row_idx, col_idx = divmod(i, num_cols)
                fig.delaxes(axes[row_idx, col_idx])

            plt.tight_layout()
            plt.show()
  
    # === Utilities ===
    def load_word2vec_model(self):
        self.word2vec_model = api.load("word2vec-google-news-300")

    def load_glove_model(self):
        self.glove_model = api.load("glove-wiki-gigaword-100")

    def load_transformer_model(self, model_name=r'C:\HuggingFace\sentence-transformers\pubmedbert-base-embeddings'):
        self.transformer_model = SentenceTransformer(model_name)

    def save_neo4j_subgraph_as_html(
            self,
            uri,
            user,
            password,
            output_path="neo4j_knowledge_graph.html",
            limit=50
        ):
            """
            Connects to a Neo4j instance and saves a subgraph visualization as HTML using PyVis.

            Parameters:
            - uri (str): Bolt+TLS URI for Neo4j Aura (e.g., neo4j+s://your-instance.databases.neo4j.io)
            - user (str): Neo4j username
            - password (str): Neo4j password
            - output_path (str): Path to save the HTML file
            - limit (int): Max number of edges to visualize
            """
            driver = GraphDatabase.driver(uri, auth=(user, password))
            net = Network(height="700px", width="100%", directed=True, cdn_resources="in_line")

            with driver.session() as session:
                result = session.run(f"""
                    MATCH (n:Concept)-[r]->(m:Concept)
                    RETURN n, r, m
                    LIMIT {limit}
                """)
                for record in result:
                    n = record["n"]
                    m = record["m"]
                    r = record["r"]

                    # Add nodes with labels and tooltips
                    net.add_node(n["cui"], label=n.get("str", n["cui"]), title=n.get("def", ""))
                    net.add_node(m["cui"], label=m.get("str", m["cui"]), title=m.get("def", ""))
                    net.add_edge(n["cui"], m["cui"], label=r.type)

            net.save_graph(output_path)
            print(f"✅ Neo4j subgraph visualization saved to: {output_path}")

    def upload_graph_to_neo4j_with_metadata(
            self,
            graph,
            tui_to_group,
            sab="MSH",
            batch_size=100,
            max_nodes=None
        ):
            """
            Uploads a NetworkX DiGraph to Neo4j, preserving node metadata and multiple edge types.
            
            Parameters:
            - graph: networkx.DiGraph
                The graph to upload (nodes = CUIs, edges with rel field)
            - tui_to_group: dict
                Mapping from TUI codes to semantic groups
            - sab: str
                Source abbreviation (default: "MSH")
            - batch_size: int
                Cypher batch size for uploads
            - max_nodes: int or None
                If set, limits the number of nodes uploaded
            """

            driver = self.driver

            # 1. Create uniqueness constraint
            with driver.session() as s:
                s.run("CREATE CONSTRAINT concept_cui IF NOT EXISTS FOR (c:Concept) REQUIRE c.cui IS UNIQUE")

            # 2. Upload nodes
            all_nodes = list(graph.nodes())
            if max_nodes is not None:
                all_nodes = all_nodes[:max_nodes]

            with driver.session() as s, tqdm(all_nodes, desc="Uploading nodes") as pbar:
                for i in range(0, len(all_nodes), batch_size):
                    batch = all_nodes[i:i+batch_size]
                    node_data = []
                    for cui in batch:
                        str_val = self.get_str(cui, sab)
                        def_val = self.get_def(cui, sab)
                        semtypes = self.get_semantic_types(cui)
                        groups = {
                            tui_to_group.get(st["TUI"], "Unknown")
                            for st in semtypes
                        }
                        node_data.append({
                            "cui": cui,
                            "str": str_val,
                            "def": def_val,
                            "groups": "; ".join(sorted(groups))
                        })
                    s.run("""
                        UNWIND $nodes AS n
                        MERGE (c:Concept {cui: n.cui})
                        SET c.str = n.str,
                            c.def = n.def,
                            c.groups = n.groups
                    """, nodes=node_data)
                    pbar.update(len(batch))

            print(f"✅ Uploaded {len(all_nodes)} nodes.")

            # 3. Upload edges with correct relationship types
            allowed = set(all_nodes)
            rel_map = {"PAR": "PARENT_OF", "RB": "BROADER_THAN", "RN": "NARROWER_THAN"}
            raw_edges = [
                (u, v, data.get("rel", "PAR"))
                for u, v, data in graph.edges(data=True)
                if u in allowed and v in allowed
            ]
            raw_edges.sort(key=lambda x: x[2])

            with driver.session() as s, tqdm(raw_edges, desc="Uploading edges") as pbar:
                for raw_rel, group in groupby(raw_edges, key=lambda x: x[2]):
                    neo_rel = rel_map.get(raw_rel, "RELATED_TO")
                    group_list = list(group)
                    for i in range(0, len(group_list), batch_size):
                        edge_batch = group_list[i:i+batch_size]
                        edge_data = [{"parent": u, "child": v} for u, v, _ in edge_batch]
                        s.run(f"""
                            UNWIND $edges AS e
                            MATCH (p:Concept {{cui: e.parent}})
                            MATCH (c:Concept {{cui: e.child}})
                            MERGE (p)-[:{neo_rel}]->(c)
                        """, edges=edge_data)
                        pbar.update(len(edge_batch))

            print(f"✅ Uploaded {len(raw_edges)} edges.")

            # 4. Indexes
            with driver.session() as s:
                s.run("CREATE INDEX concept_str IF NOT EXISTS FOR (c:Concept) ON (c.str)")
                s.run("CREATE INDEX concept_groups IF NOT EXISTS FOR (c:Concept) ON (c.groups)")

            print("🏁 Graph upload to Neo4j completed.")

    def save_concept_analysis(self, df, output_path, format="csv"):
        """
        Saves the concept analysis DataFrame to a file.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame to save
        output_path : str
            Path to the output file
        format : str
            Output format: "csv", "excel", or "parquet" (default: "csv")
        """
        if format.lower() == "csv":
            df.to_csv(output_path, index=False)
        elif format.lower() == "excel":
            df.to_excel(output_path, index=False)
        elif format.lower() == "parquet":
            df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Results saved to {output_path}")

    def create_ontology_index_table(self, sab, index_db="umls_path_cache"):
        """
        Create an ontology index table for a specific SAB (source abbreviation).
        Creates the database if it does not exist, then builds the appropriate table structure.
        """
        if not sab:
            raise ValueError("❌ Source abbreviation (SAB) must be provided.")

        table_name = f"ontology_index_{sab.lower()}"

        try:
            # Ensure index database exists
            temp_conn = mysql.connector.connect(
                host=self.mysql_info.get("host"),
                user=self.mysql_info.get("user"),
                password=self.mysql_info.get("password")
            )
            temp_cursor = temp_conn.cursor()
            temp_cursor.execute(f"CREATE DATABASE IF NOT EXISTS {index_db}")
            temp_conn.commit()
        except mysql.connector.Error as err:
            raise RuntimeError(f"❌ MySQL error while creating database '{index_db}': {err}")
        except Exception as e:
            raise RuntimeError(f"❌ Unexpected error during database creation: {e}")
        finally:
            if temp_cursor:
                temp_cursor.close()
            if temp_conn:
                temp_conn.close()

        try:
            # Now connect to the new database
            conn = mysql.connector.connect(**{**self.mysql_info, "database": index_db})
            cursor = conn.cursor()
            
            # Drop existing table if exists
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.commit()
            
            # Create new table
            cursor.execute(f"""
                CREATE TABLE {table_name} (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    CUI CHAR(8) NOT NULL,
                    DEPTH INT NOT NULL,
                    PATH TEXT NOT NULL,
                    INDEX (CUI, DEPTH)
                )
            """)
            conn.commit()

            print(f"✅ Ontology index table '{table_name}' created successfully in database '{index_db}'.")

        except mysql.connector.Error as err:
            raise RuntimeError(f"❌ MySQL error while creating table '{table_name}': {err}")
        except Exception as e:
            raise RuntimeError(f"❌ Unexpected error during table creation: {e}")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
        
    def build_and_store_mesh_closure(self, index_db="umls_path_cache", max_depth=18, batch_size=5000):
        """
        Build full MeSH closure table by precomputing shortest paths between all concept pairs.
        """
        if not hasattr(self, 'cursor') or self.cursor is None:
            raise ValueError("⚠️ No database cursor available. Are you connected?")

        table_name = "msh_closure"

        print("🔄 Building MeSH graph from MRREL...")
        try:
            G = self.build_mesh_graph_from_mrrel()
            if G.number_of_nodes() == 0:
                print("⚠️ No nodes to process. Aborting closure build.")
                return
            UG = G.to_undirected()
            print(f"✅ MeSH graph built: {UG.number_of_nodes()} nodes, {UG.number_of_edges()} edges")
        except Exception as e:
            raise RuntimeError(f"❌ Failed to build graph: {e}")

        try:
            conn = mysql.connector.connect(**{**self.mysql_info, "database": index_db})
            cursor = conn.cursor()

            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    source_cui CHAR(8) NOT NULL,
                    target_cui CHAR(8) NOT NULL,
                    path_length INT NOT NULL,
                    path_seq TEXT NOT NULL,
                    PRIMARY KEY (source_cui, target_cui),
                    INDEX idx_target (target_cui)
                )
            """)
            conn.commit()
        except mysql.connector.Error as err:
            raise RuntimeError(f"❌ MySQL error while setting up closure table: {err}")
        except Exception as e:
            raise RuntimeError(f"❌ Unexpected error setting up closure table: {e}")

        print(f"📦 Starting closure computation (up to depth {max_depth})...")
        insert_rows = []

        try:
            with tqdm(total=UG.number_of_nodes(), desc="Processing nodes", unit="node") as pbar:
                for source in UG.nodes():
                    try:
                        paths = nx.single_source_shortest_path(UG, source, cutoff=max_depth)
                        for target, path in paths.items():
                            if source == target:
                                continue
                            path_str = " ".join(path)
                            path_length = len(path) - 1
                            insert_rows.append((source, target, path_length, path_str))

                            if len(insert_rows) >= batch_size:
                                cursor.executemany(
                                    f"INSERT IGNORE INTO {table_name} (source_cui, target_cui, path_length, path_seq) VALUES (%s, %s, %s, %s)",
                                    insert_rows
                                )
                                conn.commit()
                                insert_rows.clear()
                    except Exception as e:
                        print(f"⚠️ Skipping source {source} due to error: {e}")
                    pbar.update(1)

            if insert_rows:
                cursor.executemany(
                    f"INSERT IGNORE INTO {table_name} (source_cui, target_cui, path_length, path_seq) VALUES (%s, %s, %s, %s)",
                    insert_rows
                )
                conn.commit()

        except Exception as e:
            raise RuntimeError(f"❌ Error during closure computation: {e}")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

        print("🏁 Closure build complete.")
    