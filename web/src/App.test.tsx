// Component test with a mocked API: clicking a citation chip highlights exactly
// text.slice(start_char, end_char); refusals and dropped claims are shown.
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";
import type { QueryResponse } from "./api";

const TEXT = "1. Term. This Agreement lasts two years.\n2. Law. The laws of Delaware govern.";
const start = TEXT.indexOf("The laws of Delaware");
const end = start + "The laws of Delaware".length;

function mockApi(response: QueryResponse) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      const body =
        url === "/contracts"
          ? [{ id: 7, title: "Acme Supply Agreement", split: "dev" }]
          : url === "/contracts/7"
            ? { id: 7, title: "Acme Supply Agreement", split: "dev", text: TEXT }
            : response;
      return new Response(JSON.stringify(body), { status: 200 });
    }),
  );
}

async function askQuestion() {
  const user = userEvent.setup();
  render(<App />);
  await user.selectOptions(await screen.findByLabelText("Contract", { selector: "select" }), "7");
  await screen.findByText(/This Agreement lasts/);
  await user.type(screen.getByLabelText("Question"), "Governing law?");
  await user.click(screen.getByRole("button", { name: "Ask" }));
  return user;
}

afterEach(() => vi.unstubAllGlobals());

test("clicking a chip highlights the exact cited span", async () => {
  mockApi({
    answer: "Delaware law governs.",
    claims: [
      {
        text: "Delaware law governs the agreement.",
        citations: [{ chunk_id: 3, contract_id: 7, start_char: start, end_char: end, quote: "x" }],
      },
    ],
    refused: false,
    dropped_claims: 1,
    trace_id: "abc",
  });
  const user = await askQuestion();
  await user.click(await screen.findByRole("button", { name: "Citation 1" }));
  const marks = document.querySelectorAll("mark.active");
  expect([...marks].map((m) => m.textContent).join("")).toBe(TEXT.slice(start, end));
  expect(screen.getByText(/1 claim was dropped/)).toBeInTheDocument();
});

test("a refusal is shown as a refusal", async () => {
  mockApi({
    answer: "I couldn't find support for this in the contract.",
    claims: [],
    refused: true,
    dropped_claims: 2,
    trace_id: "def",
  });
  await askQuestion();
  expect(await screen.findByText("Refused.")).toBeInTheDocument();
  expect(screen.getByText(/2 generated claims failed verification/)).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /Citation/ })).toBeNull();
});
