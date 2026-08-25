import chromadb

client_db = chromadb.PersistentClient(path="vectorstore")
client_db.delete_collection(name="mon_cours")
print("Collection supprimée avec succès")