import { createServer } from "vite";

const server = await createServer({
  logLevel: "silent",
  server: { host: "127.0.0.1" },
});

await server.listen();
const address = server.httpServer.address();
if (!address || typeof address === "string") {
  throw new Error("Vite did not bind a TCP port");
}
console.log(`GRADLAB_VITE_URL=http://127.0.0.1:${address.port}`);

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.once(signal, async () => {
    await server.close();
    process.exit(0);
  });
}
