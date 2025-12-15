import { useState, useRef } from "react";
import "./App.css";

const API_URL = "http://localhost:3001/api/chat/stream";

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const abortRef = useRef(null);

  async function sendMessage() {
    if (!input.trim() || streaming) return;

    const userMessage = { role: "user", content: input };
    setMessages((m) => [...m, userMessage, { role: "assistant", content: "" }]);
    setInput("");
    setStreaming(true);

    const controller = new AbortController();
    abortRef.current = controller;

    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: userMessage.content }),
      signal: controller.signal,
    });

    const reader = res.body.getReader();
const decoder = new TextDecoder();

let buffer = "";
let assistantText = "";

while (true) {
  const { value, done } = await reader.read();
  if (done) break;

  buffer += decoder.decode(value, { stream: true });

  const lines = buffer.split("\n");
  buffer = lines.pop(); // keep incomplete line

  for (const line of lines) {
    if (!line.startsWith("data:")) continue;

    const data = line.replace(/^data:\s*/, "");

    if (data === "done") {
      setStreaming(false);
      return;
    }

    if (assistantText.length === 0) {
  assistantText = data;
} else {
  // Add space unless data already starts with punctuation or newline
  if (/^[\s.,!?*:\n]/.test(data)) {
    assistantText += data;
  } else {
    assistantText += " " + data;
  }
}


    setMessages((msgs) => {
      const updated = [...msgs];
      updated[updated.length - 1] = {
        role: "assistant",
        content: assistantText,
      };
      return updated;
    });
  }
}
    setStreaming(false);
  }

  return (
    <div className="container">
      <h1>Training Assistant</h1>

      <div className="chat">
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <strong>{m.role === "user" ? "You" : "Assistant"}:</strong>
            <pre>{m.content}</pre>
          </div>
        ))}
      </div>

      <div className="input">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          placeholder="Ask a training question..."
        />
        <button onClick={sendMessage} disabled={streaming}>
          Send
        </button>
      </div>
    </div>
  );
}
