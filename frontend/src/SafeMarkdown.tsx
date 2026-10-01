import { Component, type ReactNode } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
class FormulaBoundary extends Component<
  { raw: string; children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? (
      <span className="formula-raw">{this.props.raw}</span>
    ) : (
      this.props.children
    );
  }
}
const mathMarkdown = (value: string) =>
  value
    .replace(/\\\[([\s\S]*?)\\\]/g, "$$$$$1$$$$")
    .replace(/\\\((.+?)\\\)/g, "$$$1$$");
export default function SafeMarkdown({ children }: { children: string }) {
  return (
    <FormulaBoundary raw={children}>
      <Markdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[
          [
            rehypeKatex,
            { throwOnError: false, strict: "ignore", trust: false },
          ],
        ]}
      >
        {mathMarkdown(children)}
      </Markdown>
    </FormulaBoundary>
  );
}
