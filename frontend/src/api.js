// src/api.js
export function streamChat(message, onEvent) {
    // EventSource only supports GET, so we use fetch with a ReadableStream instead
    const controller = new AbortController();

    fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
        signal: controller.signal,
    }).then(async (res) => {
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n\n");
            buffer = lines.pop(); // keep incomplete chunk

            for (const line of lines) {
                if (line.startsWith("data: ")) {
                    try {
                        const payload = JSON.parse(line.slice(6));
                        onEvent(payload);
                    } catch { }
                }
            }
        }
    });

    return () => controller.abort(); // return cancel function
}