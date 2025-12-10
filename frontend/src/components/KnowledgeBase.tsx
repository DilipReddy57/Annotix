import React, { useEffect, useState } from "react";
import axios from "axios";
import { Database, Search, Brain } from "lucide-react";
import { motion } from "framer-motion";

interface RagEntry {
  label: string;
  image_id: string;
}

const KnowledgeBase: React.FC = () => {
  const [entries, setEntries] = useState<RagEntry[]>([]);
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    const fetchEntries = async () => {
      try {
        const res = await axios.get("http://localhost:8000/rag/entries");
        setEntries(res.data);
      } catch (e) {
        console.error("Failed to fetch RAG entries", e);
      }
    };
    fetchEntries();
  }, []);

  const filteredEntries = entries.filter((e) =>
    e.label.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="space-y-6 h-full flex flex-col">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Brain className="text-purple-400" />
            Knowledge Base
          </h2>
          <p className="text-gray-400 text-sm">
            Visualizing {entries.length} learned concepts in RAG memory.
          </p>
        </div>
        <div className="relative">
          <Search
            className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400"
            size={18}
          />
          <input
            type="text"
            placeholder="Search concepts..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="bg-white/5 border border-white/10 rounded-lg pl-10 pr-4 py-2 text-sm focus:outline-none focus:border-purple-500 transition-colors w-64"
          />
        </div>
      </div>

      <div className="flex-1 glass rounded-xl overflow-hidden flex flex-col">
        <div className="grid grid-cols-3 bg-white/5 p-4 font-mono text-xs text-gray-400 uppercase tracking-wider border-b border-white/10">
          <div>Concept Label</div>
          <div>Source Image ID</div>
          <div>Embedding Status</div>
        </div>
        <div className="overflow-y-auto flex-1">
          {filteredEntries.map((entry, idx) => (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: idx * 0.05 }}
              key={idx}
              className="grid grid-cols-3 p-4 border-b border-white/5 hover:bg-white/5 transition-colors font-mono text-sm"
            >
              <div className="flex items-center gap-2 text-purple-300">
                <Database size={14} />
                {entry.label}
              </div>
              <div className="text-gray-500 truncate" title={entry.image_id}>
                {entry.image_id}
              </div>
              <div className="flex items-center gap-2 text-green-400 text-xs">
                <span className="w-2 h-2 rounded-full bg-green-500" />
                Indexed (512-dim)
              </div>
            </motion.div>
          ))}
          {filteredEntries.length === 0 && (
            <div className="p-8 text-center text-gray-500">
              No entries found. Process images to build knowledge.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default KnowledgeBase;
