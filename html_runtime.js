(() => {
  "use strict";

  const cells = new Map();
  const formulaCells = [];
  let calculationVersion = 0;
  let samplingChartBody = null;
  let setSamplingChartOpen = () => {};
  let hiddenInputBadges = new WeakMap();
  let selectedFormulaCell = null;
  let selectedDependencies = [];

  const key = (sheet, row, column) => `${sheet}\u0000${row}\u0000${column}`;
  const flatten = (values) => values.flatMap((value) =>
    value && value.kind === "range" ? value.values : [value]
  );
  const isError = (value) => value && value.kind === "error";
  const spreadsheetError = (message = "#N/A") => ({ kind: "error", message });

  function readStoredValue(element) {
    const rawNumber = element.getAttribute("sdval");
    if (rawNumber !== null && rawNumber !== "") {
      const number = Number(rawNumber);
      if (Number.isFinite(number)) return number;
    }
    const raw = element.getAttribute("data-sheets-value");
    if (raw) {
      try {
        const value = JSON.parse(raw);
        if (value["1"] === 3) return Number(value["3"]);
        if (value["1"] === 2) return value["2"] ?? "";
      } catch (_error) {
        // Fall through to the visible value when LibreOffice emits an odd attribute.
      }
    }
    return element.textContent.trim();
  }

  document.querySelectorAll(".worksheet").forEach((section) => {
    const sheet = section.getAttribute("aria-label");
    const columnOffset = Number(section.dataset.columnOffset || 0);
    const rowOffset = Number(section.dataset.rowOffset || 0);
    const table = section.querySelector(":scope > table");
    if (!table) return;
    [...table.rows].forEach((rowElement, rowIndex) => {
      let column = 1 + columnOffset;
      [...rowElement.cells].forEach((element) => {
        column = Number(element.dataset.gridColumn || column);
        const cell = {
          sheet,
          row: rowIndex + 1 + rowOffset,
          column,
          element,
          baseValue: readStoredValue(element),
          formula: element.getAttribute("data-sheets-formula"),
          ast: null,
          parseError: null,
          version: -1,
          evaluating: false,
          cachedValue: undefined,
        };
        cells.set(key(sheet, cell.row, column), cell);
        element.dataset.gridColumn = column;
        if (cell.formula) formulaCells.push(cell);
        column += element.colSpan || 1;
      });
    });
  });

  class FormulaParser {
    constructor(source) {
      this.source = source.startsWith("=") ? source.slice(1) : source;
      this.index = 0;
    }

    parse() {
      const expression = this.parseComparison();
      this.skipSpace();
      if (this.index !== this.source.length) {
        throw new Error(`Unexpected formula text at ${this.index}`);
      }
      return expression;
    }

    skipSpace() {
      while (/\s/.test(this.source[this.index] || "")) this.index += 1;
    }

    takeOperator(operators) {
      this.skipSpace();
      for (const operator of operators) {
        if (this.source.startsWith(operator, this.index)) {
          this.index += operator.length;
          return operator;
        }
      }
      return null;
    }

    parseComparison() {
      let node = this.parseConcat();
      while (true) {
        const operator = this.takeOperator(["<=", ">=", "<>", "=", "<", ">"]);
        if (!operator) return node;
        node = { type: "binary", operator, left: node, right: this.parseConcat() };
      }
    }

    parseConcat() {
      let node = this.parseAdditive();
      while (this.takeOperator(["&"])) {
        node = { type: "binary", operator: "&", left: node, right: this.parseAdditive() };
      }
      return node;
    }

    parseAdditive() {
      let node = this.parseMultiplicative();
      while (true) {
        const operator = this.takeOperator(["+", "-"]);
        if (!operator) return node;
        node = { type: "binary", operator, left: node, right: this.parseMultiplicative() };
      }
    }

    parseMultiplicative() {
      let node = this.parsePower();
      while (true) {
        const operator = this.takeOperator(["*", "/"]);
        if (!operator) return node;
        node = { type: "binary", operator, left: node, right: this.parsePower() };
      }
    }

    parsePower() {
      let node = this.parseUnary();
      if (this.takeOperator(["^"])) {
        node = { type: "binary", operator: "^", left: node, right: this.parsePower() };
      }
      return node;
    }

    parseUnary() {
      const operator = this.takeOperator(["+", "-"]);
      if (operator) return { type: "unary", operator, value: this.parseUnary() };
      return this.parsePrimary();
    }

    parsePrimary() {
      this.skipSpace();
      if (this.takeOperator(["("])) {
        const node = this.parseComparison();
        if (!this.takeOperator([")"])) throw new Error("Missing closing parenthesis");
        return node;
      }

      const rest = this.source.slice(this.index);
      const number = rest.match(/^(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?/);
      if (number) {
        this.index += number[0].length;
        return { type: "literal", value: Number(number[0]) };
      }
      if (rest.startsWith('"')) {
        this.index += 1;
        let value = "";
        while (this.index < this.source.length) {
          if (this.source[this.index] !== '"') {
            value += this.source[this.index++];
          } else if (this.source[this.index + 1] === '"') {
            value += '"';
            this.index += 2;
          } else {
            this.index += 1;
            return { type: "literal", value };
          }
        }
        throw new Error("Unterminated string");
      }

      const reference = rest.match(/^(?:([A-Za-z_][A-Za-z0-9_ ]*)!)?R(\[-?\d+\]|\d*)C(\[-?\d+\]|\d*)/i);
      if (reference) {
        this.index += reference[0].length;
        let node = {
          type: "reference",
          sheet: reference[1] || null,
          row: this.parseCoordinate(reference[2]),
          column: this.parseCoordinate(reference[3]),
        };
        if (this.takeOperator([":"])) {
          const end = this.parsePrimary();
          if (end.type !== "reference") throw new Error("Range must end with a reference");
          node = { type: "range", start: node, end };
        }
        return node;
      }

      const identifier = rest.match(/^([A-Za-z_][A-Za-z0-9_.]*)/);
      if (identifier) {
        this.index += identifier[0].length;
        const name = identifier[1].toUpperCase();
        if (!this.takeOperator(["("])) {
          if (name === "TRUE") return { type: "literal", value: true };
          if (name === "FALSE") return { type: "literal", value: false };
          throw new Error(`Unknown name ${name}`);
        }
        const args = [];
        if (!this.takeOperator([")"])) {
          do {
            args.push(this.parseComparison());
          } while (this.takeOperator([","]));
          if (!this.takeOperator([")"])) throw new Error(`Missing ) after ${name}`);
        }
        return { type: "call", name, args };
      }
      throw new Error(`Expected a value at ${this.index}`);
    }

    parseCoordinate(value) {
      if (value === "") return { mode: "current", value: 0 };
      if (value.startsWith("[")) return { mode: "relative", value: Number(value.slice(1, -1)) };
      return { mode: "absolute", value: Number(value) };
    }
  }

  formulaCells.forEach((cell) => {
    try {
      cell.ast = new FormulaParser(cell.formula).parse();
    } catch (error) {
      cell.parseError = error;
      console.warn(`Could not parse ${cell.sheet}!R${cell.row}C${cell.column}`, error);
    }
  });

  function coordinateValue(coordinate, current) {
    if (coordinate.mode === "absolute") return coordinate.value;
    if (coordinate.mode === "relative") return current + coordinate.value;
    return current;
  }

  function resolveReference(reference, context, inheritedSheet = null) {
    return {
      sheet: reference.sheet || inheritedSheet || context.sheet,
      row: coordinateValue(reference.row, context.row),
      column: coordinateValue(reference.column, context.column),
    };
  }

  function getCellValue(cell) {
    if (!cell) return 0;
    if (!cell.formula || !cell.ast) return cell.baseValue;
    if (cell.version === calculationVersion) return cell.cachedValue;
    if (cell.evaluating) return spreadsheetError("#CYCLE!");
    cell.evaluating = true;
    let value;
    try {
      value = evaluate(cell.ast, cell);
    } catch (_error) {
      value = spreadsheetError("#VALUE!");
    }
    cell.evaluating = false;
    cell.version = calculationVersion;
    cell.cachedValue = value;
    return value;
  }

  function toNumber(value) {
    if (isError(value)) return value;
    if (value && value.kind === "range") value = value.values[0] ?? 0;
    if (value === "" || value === null || value === undefined) return 0;
    if (typeof value === "boolean") return value ? 1 : 0;
    const number = Number(value);
    return Number.isFinite(number) ? number : spreadsheetError("#VALUE!");
  }

  function compare(left, right, operator) {
    const leftNumber = toNumber(left);
    const rightNumber = toNumber(right);
    const numeric = !isError(leftNumber) && !isError(rightNumber);
    const a = numeric ? leftNumber : String(left);
    const b = numeric ? rightNumber : String(right);
    if (operator === "=") return a === b;
    if (operator === "<>") return a !== b;
    if (operator === "<") return a < b;
    if (operator === ">") return a > b;
    if (operator === "<=") return a <= b;
    return a >= b;
  }

  function evaluate(node, context) {
    if (node.type === "literal") return node.value;
    if (node.type === "reference") {
      const address = resolveReference(node, context);
      return getCellValue(cells.get(key(address.sheet, address.row, address.column)));
    }
    if (node.type === "range") {
      const start = resolveReference(node.start, context);
      const end = resolveReference(node.end, context, start.sheet);
      const values = [];
      const rowStep = start.row <= end.row ? 1 : -1;
      const columnStep = start.column <= end.column ? 1 : -1;
      for (let row = start.row; ; row += rowStep) {
        for (let column = start.column; ; column += columnStep) {
          values.push(getCellValue(cells.get(key(start.sheet, row, column))));
          if (column === end.column) break;
        }
        if (row === end.row) break;
      }
      return {
        kind: "range",
        values,
        rows: Math.abs(end.row - start.row) + 1,
        columns: Math.abs(end.column - start.column) + 1,
      };
    }
    if (node.type === "unary") {
      const value = toNumber(evaluate(node.value, context));
      if (isError(value)) return value;
      return node.operator === "-" ? -value : value;
    }
    if (node.type === "binary") {
      const left = evaluate(node.left, context);
      if (isError(left)) return left;
      const right = evaluate(node.right, context);
      if (isError(right)) return right;
      if (["=", "<>", "<", ">", "<=", ">="].includes(node.operator)) {
        return compare(left, right, node.operator);
      }
      if (node.operator === "&") return String(left) + String(right);
      const a = toNumber(left);
      const b = toNumber(right);
      if (isError(a)) return a;
      if (isError(b)) return b;
      if (node.operator === "+") return a + b;
      if (node.operator === "-") return a - b;
      if (node.operator === "*") return a * b;
      if (node.operator === "/") return b === 0 ? spreadsheetError("#DIV/0!") : a / b;
      return Math.pow(a, b);
    }
    if (node.type === "call") return evaluateCall(node, context);
    return spreadsheetError("#VALUE!");
  }

  function erf(value) {
    const sign = value < 0 ? -1 : 1;
    const x = Math.abs(value);
    const t = 1 / (1 + 0.3275911 * x);
    const polynomial = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
      - 0.284496736) * t + 0.254829592) * t;
    return sign * (1 - polynomial * Math.exp(-x * x));
  }

  function evaluateCall(node, context) {
    if (node.name === "IF") {
      const condition = evaluate(node.args[0], context);
      if (isError(condition)) return condition;
      return evaluate(condition ? node.args[1] : (node.args[2] || { type: "literal", value: false }), context);
    }
    const args = node.args.map((argument) => evaluate(argument, context));
    const firstError = flatten(args).find(isError);
    if (firstError) return firstError;
    const numbers = () => flatten(args).map(toNumber).filter((value) => !isError(value));

    if (node.name === "EXACT") return String(args[0]) === String(args[1]);
    if (node.name === "NA") return spreadsheetError("#N/A");
    if (node.name === "MAX") return Math.max(...numbers());
    if (node.name === "MIN") return Math.min(...numbers());
    if (node.name === "SQRT") return Math.sqrt(toNumber(args[0]));
    if (node.name === "EXP") return Math.exp(toNumber(args[0]));
    if (node.name === "ERF") return erf(toNumber(args[0]));
    if (node.name === "LN") return Math.log(toNumber(args[0]));
    if (node.name === "ABS") return Math.abs(toNumber(args[0]));
    if (node.name === "SIN") return Math.sin(toNumber(args[0]));
    if (node.name === "MOD") {
      const a = toNumber(args[0]);
      const b = toNumber(args[1]);
      return a - b * Math.floor(a / b);
    }
    if (node.name === "COUNT") return flatten(args).filter((value) => typeof value === "number" && Number.isFinite(value)).length;
    if (node.name === "ROWS") return args[0]?.kind === "range" ? args[0].rows : 1;
    if (node.name === "LARGE") {
      const values = flatten([args[0]]).map(toNumber).filter((value) => !isError(value)).sort((a, b) => b - a);
      return values[Math.max(0, Math.trunc(toNumber(args[1])) - 1)] ?? spreadsheetError("#NUM!");
    }
    if (node.name === "MATCH") {
      const target = args[0];
      const values = flatten([args[1]]);
      const position = values.findIndex((value) => compare(value, target, "="));
      return position < 0 ? spreadsheetError("#N/A") : position + 1;
    }
    return spreadsheetError("#NAME?");
  }

  function numberFormat(cell) {
    const raw = cell.element.getAttribute("data-sheets-numberformat");
    if (raw) {
      try { return JSON.parse(raw)["2"] || ""; } catch (_error) { /* ignore */ }
    }
    return cell.element.getAttribute("sdnum") || "";
  }

  function formatValue(cell, value) {
    if (isError(value)) return value.message;
    if (document.documentElement.hasAttribute('data-training-replay') &&
        (cell.element.classList.contains('live-input-token') || cell.element.classList.contains('training-output-character'))) {
      return value === '␣' ? ' ' : String(value ?? '');
    }
    if (cell.element.classList.contains("live-input-token")) return value === " " ? "␣" : String(value ?? "");
    if (typeof value === "boolean") return value ? "TRUE" : "FALSE";
    if (typeof value !== "number" || !Number.isFinite(value)) return String(value ?? "");
    if (cell.element.classList.contains('matrix-value')) {
      if (cell.element.dataset.matrixName === 'Top-k eligibility') {
        const logits=[...cells.values()].filter(c=>c.element.dataset.matrixName==='Logits' && c.element.dataset.matrixRow===cell.element.dataset.matrixRow);
        const logit=logits.sort((a,b)=>b.column-a.column)[0];
        return logit ? Number(getCellValue(logit)).toFixed(2) : '';
      }
      if (cell.element.dataset.matrixName === 'Selected vocabulary index · 1-based') return String(Math.round(value));
      if (cell.element.dataset.matrixName === 'Temperature · Top-k · Seed' && Number(cell.element.dataset.matrixRow) === 3) return String(Math.round(value));
      return (Math.abs(value) < .005 ? 0 : value).toFixed(2);
    }
    const format = numberFormat(cell);
    if (format.includes("#,##0")) return Math.round(value).toLocaleString("en-US");
    if (/(^|;)0(;|$)/.test(format) && !format.includes("#")) return String(Math.round(value));
    const fixed = format.match(/0\.(0{2,})/);
    if (fixed && !format.includes("#")) return value.toFixed(fixed[1].length);
    if (Math.abs(value) < 0.0005) return "0";
    return Number(value.toFixed(3)).toString();
  }

  const renderedValues = new WeakMap();
  const invalidRenderedCells = new WeakSet();
  const changeAnimations = new WeakMap();
  let deferTrainingPaint = false;
  const pendingTrainingPaint = new Map();
  function queueTrainingPaint(cell) {
    if(!pendingTrainingPaint.has(cell.paintCard))pendingTrainingPaint.set(cell.paintCard,new Set());
    pendingTrainingPaint.get(cell.paintCard).add(cell);
  }
  function paintText(cell,value) {
    const target=cell.displayTarget?.isConnected?cell.displayTarget:(cell.displayTarget=cell.element.querySelector('font')||cell.element);
    const text=formatValue(cell,value);
    if(target.textContent!==text)target.textContent=text;
  }
  function paintColor(cell) {
    if(cell.element.style.backgroundColor!==cell.paintColor)cell.element.style.backgroundColor=cell.paintColor;
    if(cell.element.dataset.colorMin!==String(cell.paintMin))cell.element.dataset.colorMin=cell.paintMin;
    if(cell.element.dataset.colorMax!==String(cell.paintMax))cell.element.dataset.colorMax=cell.paintMax;
  }
  function flushTrainingPaint() {
    if(!pendingTrainingPaint.size)return;
    const visible=window.LittleLlamaWorkbook?.visibleStageCards?.();
    const ready=[];
    for(const [card,items] of pendingTrainingPaint){
      const rect=visible?null:card.getBoundingClientRect();
      if(visible?visible.has(card):rect.bottom>0&&rect.top<innerHeight)ready.push([card,items]);
    }
    for(const [card,items] of ready){
      for(const cell of items){
        if(cell.textDirty){paintText(cell,cell.displayValue);cell.textDirty=false;}
        if(cell.colorDirty){paintColor(cell);cell.colorDirty=false;}
      }
      pendingTrainingPaint.delete(card);
    }
  }
  // The annotation frame identifies visible stages before any browser paint.
  window.addEventListener('latte:annotations',flushTrainingPaint);
  function writeCell(cell, value) {
    if(deferTrainingPaint&&cell.paintCard){
      if(!Object.is(renderedValues.get(cell),value)||invalidRenderedCells.has(cell)){
        cell.displayValue=value;cell.textDirty=true;queueTrainingPaint(cell);
      }
    }else paintText(cell,value);
    // A temporary empty/invalid token must not erase the last valid baseline.
    // Compare its replacement against that baseline, while errors stay unlit.
    if (!isError(value)) renderedValues.set(cell, value);
    if (isError(value)) invalidRenderedCells.add(cell);
    else invalidRenderedCells.delete(cell);
  }
  const displayToken = value => value === ' ' || value === '<space>' ? '␣' : String(value);
  const inputToken = value => value === '␣' ? ' ' : value;

  function recalculate(highlightChanges = false) {
    const changedCells = [];
    calculationVersion += 1;
    formulaCells.forEach(cell => {
      // Each recalculation owns one highlight set. Do not leave animations
      // from an earlier token edit mixed with the latest changed cells.
      changeAnimations.get(cell.element)?.cancel();
      changeAnimations.delete(cell.element);
      const value = getCellValue(cell);
      const comparable = isError(value) ? value.message : value;
      if (highlightChanges && !isError(value) && renderedValues.has(cell) &&
          (invalidRenderedCells.has(cell) ||
           !Object.is(renderedValues.get(cell), comparable))) changedCells.push(cell);
      writeCell(cell, value);
    });
    refreshMatrixColors();
    refreshSamplingLayout();
    flushTrainingPaint();
    const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
    changedCells.forEach(cell => {
      const element = cell.element;
      if (!element.isConnected || !element.classList.contains('matrix-value')) return;
      changeAnimations.get(element)?.cancel();
      const animation = element.animate([
        {boxShadow: 'inset 0 0 0 2px #000000', offset: 0},
        {boxShadow: 'inset 0 0 0 2px #000000', offset: .65},
        {boxShadow: 'inset 0 0 0 0 transparent', offset: 1},
      ], {duration: reducedMotion ? 1800 : 2800, easing: 'ease-out'});
      changeAnimations.set(element, animation);
    });
    if (selectedFormulaCell) renderMathPanel(selectedFormulaCell);
    document.dispatchEvent(new CustomEvent("latte:recalculated"));
  }

  const characterHeaderStyle = document.createElement('style');
  characterHeaderStyle.textContent = '.matrix-layout td.live-input-token,.matrix-layout td.training-output-character{display:flex!important;align-items:center!important;justify-content:center!important;text-align:center!important;box-sizing:border-box;border:1px solid #879bb4!important;border-radius:3px;background:#edf2f9!important;padding:0!important;min-width:0;height:24px!important;align-self:center;white-space:pre!important;font:600 11px/1 ui-monospace,monospace!important}.matrix-layout td.live-input-token font,.matrix-layout td.training-output-character font{display:block!important;text-align:center!important;width:100%;white-space:pre!important}';
  document.head.append(characterHeaderStyle);
  const editableCells = new Map();
  formulaCells.forEach((cell) => {
    const match = cell.formula.match(/EXACT\(R(\d+)C,Embeddings!R\d+C1\)/i);
    if (!match) return;
    const input = cells.get(key(cell.sheet, Number(match[1]), cell.column));
    if (input) editableCells.set(key(input.sheet, input.row, input.column), input);
  });

  editableCells.forEach((cell) => {
    const element = cell.element;
    element.contentEditable = "true";
    element.spellcheck = false;
    element.classList.add("live-input-token");
    element.textContent = displayToken(cell.baseValue);
    element.setAttribute("aria-label", `Editable input token ${cell.column}`);
    element.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        element.blur();
      }
    });
    let timer;
    element.addEventListener("input", () => {
      clearTimeout(timer);
      cell.baseValue = inputToken(element.textContent.replace(/[\r\n]+/g, ""));
      const visible = displayToken(cell.baseValue);
      if (element.textContent !== visible) {
        element.textContent = visible;
        if (document.activeElement === element) {
          const range = document.createRange();
          range.selectNodeContents(element); range.collapse(false);
          const selection = window.getSelection();
          selection.removeAllRanges(); selection.addRange(range);
        }
      }
      timer = setTimeout(() => recalculate(true), 80);
    });
  });

  function columnLabel(column) {
    let label = "";
    while (column > 0) {
      column -= 1;
      label = String.fromCharCode(65 + (column % 26)) + label;
      column = Math.floor(column / 26);
    }
    return label;
  }

  function addressLabel(sheet, row, column, currentSheet = null) {
    const address = `${columnLabel(column)}${row}`;
    return sheet === currentSheet ? address : `${sheet}!${address}`;
  }

  function plainValue(value) {
    if (isError(value)) return value.message;
    if (value && value.kind === "range") {
      const numeric = value.values.filter((item) => typeof item === "number" && Number.isFinite(item));
      if (!numeric.length) return `${value.values.length} cells`;
      return `${value.values.length} cells · min ${Number(Math.min(...numeric).toFixed(3))} · max ${Number(Math.max(...numeric).toFixed(3))}`;
    }
    if (typeof value === "number") return Number(value.toFixed(6)).toString();
    return String(value ?? "");
  }

  function collectDependencies(node, context, output, seen) {
    if (!node) return;
    if (node.type === "reference") {
      const address = resolveReference(node, context);
      const referenceKey = key(address.sheet, address.row, address.column);
      if (!seen.has(referenceKey)) {
        seen.add(referenceKey);
        const dependency = cells.get(referenceKey);
        output.push({
          label: addressLabel(address.sheet, address.row, address.column, context.sheet),
          value: getCellValue(dependency),
          elements: dependency ? [dependency.element] : [],
        });
      }
      return;
    }
    if (node.type === "range") {
      const start = resolveReference(node.start, context);
      const end = resolveReference(node.end, context, start.sheet);
      const rangeKey = `${start.sheet}:${start.row}:${start.column}:${end.row}:${end.column}`;
      if (!seen.has(rangeKey)) {
        seen.add(rangeKey);
        const rangeValue = evaluate(node, context);
        const elements = [];
        const rowStep = start.row <= end.row ? 1 : -1;
        const columnStep = start.column <= end.column ? 1 : -1;
        for (let row = start.row; ; row += rowStep) {
          for (let column = start.column; ; column += columnStep) {
            const dependency = cells.get(key(start.sheet, row, column));
            if (dependency && elements.length < 200) elements.push(dependency.element);
            if (column === end.column) break;
          }
          if (row === end.row) break;
        }
        output.push({
          label: `${addressLabel(start.sheet, start.row, start.column, context.sheet)}:${addressLabel(end.sheet, end.row, end.column, context.sheet)}`,
          value: rangeValue,
          elements,
        });
      }
      return;
    }
    if (node.type === "binary") {
      collectDependencies(node.left, context, output, seen);
      collectDependencies(node.right, context, output, seen);
    } else if (node.type === "unary") {
      collectDependencies(node.value, context, output, seen);
    } else if (node.type === "call") {
      node.args.forEach((argument) => collectDependencies(argument, context, output, seen));
    }
  }

  function formulaExplanation(formula) {
    if (/EXACT\(.*Embeddings!/i.test(formula)) return "Looks up the learned embedding vector for the selected input token.";
    if (/ERF\(/i.test(formula)) return "Applies the GELU activation: the error function smoothly gates each value instead of clipping it.";
    if (/EXP\(/i.test(formula) && /MAX\(/i.test(formula)) return "Computes a numerically stable softmax, turning scores or logits into normalized weights.";
    if (/SQRT\(/i.test(formula) && /\^2/.test(formula)) return "Performs RMS normalization by scaling the vector with its root-mean-square magnitude.";
    if (/LN\(/i.test(formula) && /SIN\(/i.test(formula)) return "Creates deterministic sampling noise from the seed and converts it to a Gumbel score.";
    if (/MATCH\(/i.test(formula)) return "Selects the vocabulary entry with the largest final sampling score.";
    if (/COUNT\(|ROWS\(/i.test(formula)) return "Counts the model parameters and matrix dimensions represented by this workbook.";
    if (/\*.*\+|\+.*\*/.test(formula)) return "Combines upstream values with matrix/vector arithmetic for this stage of the forward pass.";
    if (/\+/.test(formula)) return "Adds upstream values, such as a residual connection or token-plus-position embedding.";
    return "Evaluates this workbook formula using the currently calculated upstream cells.";
  }

  function mathematicalFormula(cell) {
    const operation = cell.element.closest('.stage-card')?.dataset.operation || '';
    const formula = cell.formula;
    if (/EXACT\(.*Embeddings!/i.test(formula)) return 'eᵢ = E[tokenᵢ]';
    if (/Token \+ position/i.test(operation)) {
      if (/SIN\(/i.test(formula)) return 'p(pos, 2j) = sin(pos / 10000^(2j/d))';
      if (/COS\(/i.test(formula)) return 'p(pos, 2j+1) = cos(pos / 10000^(2j/d))';
      return 'xᵢ = eᵢ + pᵢ';
    }
    if (/RMSNorm/i.test(operation)) return 'RMSNorm(x) = γ ⊙ x / √(mean(x²) + ε)';
    if (/Key projection/i.test(operation)) return 'K = WₖX';
    if (/Query projection/i.test(operation)) return 'Q = WᵩX';
    if (/Value projection/i.test(operation)) return 'V = WᵥX';
    if (/Scores/i.test(operation)) return 'S = KᵀQ / √dₖ';
    if (/Causal mask|softmax/i.test(operation) && !/Vocabulary/i.test(operation)) return 'Aᵢⱼ = exp(Sᵢⱼ + Mᵢⱼ) / Σₖ exp(Sₖⱼ + Mₖⱼ)';
    if (/weighted value/i.test(operation)) return 'H = VA';
    if (/Concatenate/i.test(operation)) return 'H = concat(H₁, …, Hₕ)';
    if (/Output projection WO/i.test(operation)) return 'Y = WₒH';
    if (/RESIDUAL/i.test(operation)) return 'y = x + f(x)';
    if (/GELU/i.test(operation) || /ERF\(/i.test(formula)) return 'GELU(x) = ½x[1 + erf(x / √2)]';
    if (/MLP.*Input projection/i.test(operation)) return 'z = W₁x + b₁';
    if (/MLP.*Output projection/i.test(operation)) return 'y = W₂ GELU(z) + b₂';
    if (/Vocabulary/i.test(operation)) return /EXP\(/i.test(formula)
      ? 'pᵢ = exp(ℓᵢ − max(ℓ)) / Σⱼ exp(ℓⱼ − max(ℓ))'
      : 'ℓ = Wᵥₒ꜀ₐᵦx';
    if (/Gumbel|LN\(/i.test(formula)) return 'gᵢ = −ln(−ln(uᵢ)),  uᵢ ∈ (0, 1)';
    if (/sampling/i.test(operation)) return 'token = argmaxᵢ∈top-k (ℓᵢ / τ + gᵢ)';
    if (/COUNT\(|ROWS\(/i.test(formula)) return 'N = Σ number of learned parameters';
    return 'y = f(x)';
  }

  const gridNames = new Map();
  const matrixColorGroups = [];

  // Independent ranges preserve small-value patterns without changing values.
  const cachePalette={weight:[242,163,104],query:[241,215,83],key:[138,219,139],value:[223,157,215],attention:[0,181,226],output:[173,176,180]};
  function refreshMatrixColors() {
    matrixColorGroups.forEach(group => {
      const {component,weight}=group;
      if (component[0]?.element.dataset.matrixName === 'Temperature · Top-k · Seed') {
        component.forEach(cell => {
          cell.element.style.backgroundColor = '#328359';
          cell.element.style.color = '#ffffff';
          delete cell.element.dataset.colorMin;
          delete cell.element.dataset.colorMax;
        });
        return;
      }
      const values = component.map(cell => getCellValue(cell));
      if(deferTrainingPaint&&group.colorValues&&values.every((value,index)=>Object.is(value,group.colorValues[index])))return;
      group.colorValues=values;
      const roleColor=window.LatteMath?.roleColors[window.LatteMath.matrixRole(component[0]?.element.dataset.matrixName||'',weight)];
      let minimum = Infinity, maximum = -Infinity;
      values.forEach(value => {
        if (typeof value === 'number' && Number.isFinite(value)) {
          minimum = Math.min(minimum,value);
          maximum = Math.max(maximum,value);
        }
      });
      component.forEach((cell,index) => {
        const value = values[index];
        if (typeof value !== 'number' || !Number.isFinite(value)) return;
        let intensity;
        const base=cachePalette[cell.element.dataset.kvColor] || roleColor;
        const neutral = base ? base.map(channel=>Math.round(255*.88+channel*.12)) : weight ? [255,246,185] : [226,232,240];
        const target = base || (weight ? [246,172,55] : value < 0 ? [93,143,205] : [238,104,97]);
        if (weight) intensity = maximum === minimum ? (value === 0 ? 0 : .65) : (value-minimum)/(maximum-minimum);
        else if (minimum < 0 && maximum > 0) intensity = value < 0 ? value/minimum : value/maximum;
        else if (maximum === minimum) intensity = value === 0 ? 0 : .65;
        else intensity = maximum <= 0 ? (maximum-value)/(maximum-minimum) : (value-minimum)/(maximum-minimum);
        const color = `rgb(${neutral.map((channel,i) => Math.round(channel+(target[i]-channel)*intensity)).join(',')})`;
        const dirty=cell.paintColor!==color||cell.paintMin!==minimum||cell.paintMax!==maximum;
        cell.paintColor=color;cell.paintMin=minimum;cell.paintMax=maximum;
        if(deferTrainingPaint&&cell.paintCard){
          if(dirty){cell.colorDirty=true;queueTrainingPaint(cell);}
        }else paintColor(cell);
      });
    });
  }

  function indexMatrices() {
    const sheetRows = new Map();
    cells.forEach(cell => {
      const rowKey = `${cell.sheet}:${cell.row}`;
      if (!sheetRows.has(rowKey)) sheetRows.set(rowKey, []);
      sheetRows.get(rowKey).push(cell);
    });
    const numeric = new Map([...cells].filter(([, cell]) => cell.element.hasAttribute('sdval')));
    const visited = new Set();
    const nameCounts = new Map();
    numeric.forEach((seed, seedKey) => {
      if (visited.has(seedKey)) return;
      const queue = [seed];
      const component = [];
      visited.add(seedKey);
      for (let index = 0; index < queue.length; index++) {
        const cell = queue[index];
        component.push(cell);
        [[-1,0],[1,0],[0,-1],[0,1]].forEach(([dr, dc]) => {
          const nextKey = key(cell.sheet, cell.row + dr, cell.column + dc);
          const next = numeric.get(nextKey);
          if (next && !visited.has(nextKey)) { visited.add(nextKey); queue.push(next); }
        });
      }
      const top = Math.min(...component.map(cell => cell.row));
      const left = Math.min(...component.map(cell => cell.column));
      const right = Math.max(...component.map(cell => cell.column));
      let label = '';
      for (let row = top - 1; row >= Math.max(1, top - 4) && !label; row--) {
        const candidates = (sheetRows.get(`${seed.sheet}:${row}`) || []).filter(cell =>
          cell.column >= left - 1 && cell.column <= right && !cell.formula &&
          !cell.element.classList.contains('live-input-token') &&
          typeof cell.baseValue === 'string' && cell.baseValue.trim() &&
          !/^LAYER\s|^\[|^[↓→↑←]$/.test(cell.baseValue.trim()));
        label = candidates.sort((a,b) => Math.abs(a.column - left) - Math.abs(b.column - left))[0]?.baseValue.trim() || '';
      }
      label ||= seed.element.closest('.stage-card')?.dataset.operation || seed.sheet;
      const scope = `${seed.sheet}:${label}`;
      const serial = (nameCounts.get(scope) || 0) + 1;
      nameCounts.set(scope, serial);
      const name = serial === 1 ? label : `${label} · matrix ${serial}`;
      const originalColor = seed.element.getAttribute('bgcolor') || '';
      const channels = /^#[0-9a-f]{6}$/i.test(originalColor) ? [1,3,5].map(index => parseInt(originalColor.slice(index,index+2),16)) : null;
      const weight = !!channels && channels[0] > channels[2]+25 && channels[1] > channels[2]+25;
      matrixColorGroups.push({component,weight});
      const multipleRows=component.some(cell=>cell.row!==top);
      component.forEach(cell => {
        cell.paintCard=cell.element.closest('.stage-card');
        const row = cell.row - top + 1;
        const column = cell.column - left + 1;
        gridNames.set(key(cell.sheet, cell.row, cell.column), `${name}[${row-1},${column-1}]`);
        cell.element.dataset.matrixName = name;
        cell.element.classList.toggle('learned-parameter', weight);
        if (name === 'Predicted next character') cell.element.classList.add('training-output-character');
        cell.element.dataset.matrixRow = row;
        cell.element.dataset.matrixColumn = column;
        cell.element.title = `${name} · row ${row-1}, column ${column-1}`;
        if (cell.row === top && right > left) cell.element.dataset.columnIndex = column-1;
        else delete cell.element.dataset.columnIndex;
        if (cell.column === left) {
          if (multipleRows && !['Sampling noise · saved seeded draw','Sampling score','Top-k candidates','Top-k probabilities'].includes(label)) cell.element.dataset.rowIndex = row-1;
          if (/^(Logits|Vocabulary probabilities|Top-k eligibility)$/.test(label)) {
            const vocabularyCell = cells.get(key('Embeddings', row + 5, 1));
            if (vocabularyCell) {
              const token = String(getCellValue(vocabularyCell));
              const visibleToken = token === ' ' || token === '<space>' ? '␣'
                : token === '\n' || token === '\\n' ? '\\n'
                : token === '\t' || token === '\\t' ? '\\t' : token;
              const tokenLabel = document.createElement('span');
              tokenLabel.className = 'vocabulary-token-label';
              tokenLabel.textContent = visibleToken;
              cell.element.appendChild(tokenLabel);
              cell.element.classList.add('vocabulary-row-label');
              cell.element.setAttribute('aria-label', `${name}, vocabulary token ${visibleToken}, row ${row-1}`);
            }
          }
        }
      });
    });
  }

  function gridLabel(cell) {
    if (!cell) return 'empty input';
    return gridNames.get(key(cell.sheet, cell.row, cell.column)) ||
      (cell.element.classList.contains('live-input-token') ? `Token[${[...editableCells.values()].indexOf(cell)}]` :
        `${cell.sheet} · ${String(cell.baseValue).slice(0, 45) || 'constant'}`);
  }

  function calculationExpression(node, context, numeric = false, references = null) {
    if (!node) return 'unavailable';
    if (node.type === 'literal') return typeof node.value === 'string' ? JSON.stringify(node.value) : String(node.value);
    if (node.type === 'reference') {
      const address = resolveReference(node, context);
      const source = cells.get(key(address.sheet, address.row, address.column));
      if (source && references) references.set(key(source.sheet, source.row, source.column), source);
      return numeric ? plainValue(getCellValue(source)) : gridLabel(source);
    }
    if (node.type === 'range') {
      const start = resolveReference(node.start, context);
      const end = resolveReference(node.end, context, start.sheet);
      if (numeric) return `[${evaluate(node, context).values.map(plainValue).join(', ')}]`;
      const first = cells.get(key(start.sheet, start.row, start.column));
      const last = cells.get(key(end.sheet, end.row, end.column));
      if (references) {
        for (let row = Math.min(start.row,end.row); row <= Math.max(start.row,end.row); row++) {
          for (let column = Math.min(start.column,end.column); column <= Math.max(start.column,end.column); column++) {
            const source = cells.get(key(start.sheet,row,column));
            if (source) references.set(key(source.sheet,row,column), source);
          }
        }
      }
      return `${gridLabel(first)} … ${gridLabel(last)}`;
    }
    if (node.type === 'unary') return `${node.operator}(${calculationExpression(node.value,context,numeric,references)})`;
    if (node.type === 'binary') {
      const operator = {'*':'×','/':'÷','<>':'≠','&':'concat'}[node.operator] || node.operator;
      return `(${calculationExpression(node.left,context,numeric,references)} ${operator} ${calculationExpression(node.right,context,numeric,references)})`;
    }
    if (node.type === 'call') {
      if (node.name === 'IF') {
        const branch = evaluate(node.args[0],context) ? node.args[1] : node.args[2];
        return calculationExpression(branch,context,numeric,references);
      }
      const names = {SUM:'sum',SUMPRODUCT:'dot product',SQRT:'√',EXP:'exp',MAX:'max',MIN:'min',AVERAGE:'mean',EXACT:'equals',LN:'ln',ERF:'erf',SIN:'sin',COS:'cos',POWER:'power',ABS:'abs',MATCH:'matching index',INDEX:'select',COUNT:'count',ROWS:'row count',NA:'undefined'};
      return `${names[node.name] || node.name.toLowerCase()}(${node.args.map(arg => calculationExpression(arg,context,numeric,references)).join(', ')})`;
    }
    return 'unavailable';
  }

  function renderCalculationDetails(cell) {
    const references = new Map();
    const symbolic = calculationExpression(cell.ast,cell,false,references);
    const substituted = calculationExpression(cell.ast,cell,true);
    mathPanel.querySelector('.math-equation').textContent = `${gridLabel(cell)} = ${symbolic}`;
    mathPanel.querySelector('.math-substitution').textContent = `${substituted} = ${plainValue(getCellValue(cell))}`;
    const inputs = mathPanel.querySelector('.math-grid-inputs');
    inputs.replaceChildren(...[...references.values()].map(source => {
      const button = document.createElement('button');
      button.className = 'grid-input-term';
      button.textContent = `${gridLabel(source)} = ${plainValue(getCellValue(source))}`;
      button.addEventListener('click', () => {
        const block = blocks.find(candidate => candidate.rows.includes(source.element.closest('tr')));
        if (block) {
          if (focusSnapshot) restoreFocus();
          block.group.open = true;
          setBlockCollapsed(block,false);
        } else {
          const section = source.element.closest('.worksheet');
          document.querySelector(`.sheet-tab[data-sheet="${section?.id}"]`)?.click();
        }
        source.element.scrollIntoView({behavior:'smooth',block:'center',inline:'center'});
        document.querySelectorAll('.math-term-selected').forEach(element => element.classList.remove('math-term-selected'));
        source.element.classList.add('math-term-selected');
      });
      return button;
    }));
  }

  const mathPanel = document.createElement("aside");
  mathPanel.className = "math-panel";
  mathPanel.setAttribute("aria-label", "Math explanation");
  mathPanel.innerHTML = `
    <div class="math-panel-header"><div><h2>Math explanation</h2><div class="math-address"></div></div><button class="math-close" type="button" aria-label="Close">×</button></div>
    <div class="math-result"></div>
    <p class="math-summary"></p>
    <h3>Calculation · row, column</h3><div class="math-equation"></div>
    <h3>Substitute current values</h3><div class="math-substitution"></div>
    <h3>Inputs · click to locate</h3><div class="math-grid-inputs"></div>
    <p class="math-rounding-note">Values are rounded for display. Calculations use full precision.</p>`;
  // Math explainer temporarily disabled pending redesign.
  // document.body.appendChild(mathPanel);
  const inputMatrixCard = document.createElement('aside');
  inputMatrixCard.className = 'input-matrix-card';
  inputMatrixCard.hidden = true;
  inputMatrixCard.setAttribute('aria-label', 'Input matrices for selected cell');
  document.body.append(inputMatrixCard);
  const inputReturnCard = document.createElement('aside');
  inputReturnCard.className = 'input-matrix-card input-return-card';
  inputReturnCard.setAttribute('aria-label', 'Return from input cells');
  inputReturnCard.hidden = true;
  document.body.append(inputReturnCard);
  const clearSelectionButton = document.createElement('button');
  clearSelectionButton.type = 'button';
  clearSelectionButton.className = 'toolbar-button selection-clear';
  clearSelectionButton.textContent = 'Clear cell focus';
  clearSelectionButton.hidden = true;
  clearSelectionButton.addEventListener('click', closeMathPanel);
  document.body.append(clearSelectionButton);
  let returnCardElements = [];
  function positionReturnCard() {
    if (inputReturnCard.hidden || !returnCardElements.length) return;
    const rects = returnCardElements.map(element => element.getBoundingClientRect());
    const right = Math.max(...rects.map(rect => rect.right));
    const top = Math.min(...rects.map(rect => rect.top));
    const bottom = Math.max(...rects.map(rect => rect.bottom));
    const box = inputReturnCard.getBoundingClientRect();
    inputReturnCard.style.left = `${right + 10}px`;
    inputReturnCard.style.top = `${(top+bottom-box.height)/2}px`;
  }
  let inputCardCell = null;
  function positionInputCard() {
    if (!inputCardCell || inputMatrixCard.hidden) return;
    const rect = inputCardCell.element.getBoundingClientRect();
    const right = rect.right + 10;
    inputMatrixCard.style.left = `${right}px`;
    inputMatrixCard.style.top = `${rect.top}px`;
  }
  function compactMatrixName(name) {
    if(/^W[KQV] ·/.test(name))return 'W'+name[1].toLowerCase();
    if(/^x ·/.test(name))return 'x';
    if(/^Scores ·/.test(name))return 'Scores';
    if(/^VA ·/.test(name))return 'VA';
    if(/^A ·/.test(name))return 'A';
    if(/^Kᵀ ·/.test(name))return 'Kᵀ';
    if(/^[KQV] ·/.test(name))return name[0];
    return name;
  }
  function showInputMatrices(cell) {
    hiddenInputBadges = new WeakMap();
    clearSelectionButton.hidden = false;
    inputReturnCard.hidden = true;
    returnCardElements = [];
    inputCardCell = cell;
    const references = new Map();
    if (cell.ast) calculationExpression(cell.ast, cell, false, references);
    if (cell.element.dataset.matrixName === 'Top-k eligibility') {
      references.clear();
      const source=displayedLogitSource(cell);
      if(source)references.set(key(source.sheet,source.row,source.column),source);
    }
    if (cell.element.dataset.matrixName === 'Top-k candidates') {
      references.clear();
      const source=[...cells.values()].find(c=>c.element.dataset.matrixName==='Top-k eligibility' && c.element.dataset.matrixRow===cell.element.dataset.matrixRow);
      if(source)references.set(key(source.sheet,source.row,source.column),source);
    }
    if (cell.element.dataset.matrixName === 'Top-k probabilities') {
      references.clear();
      temperatureSources().forEach(source=>references.set(key(source.sheet,source.row,source.column),source));
    }
    const sources = new Map();
    references.forEach(source => {
      if (source.element.style.display === 'none') return;
      const name = source.element.dataset.matrixName || (source.element.classList.contains('live-input-token') ? 'Input sequence' : source.sheet);
      if (!sources.has(name)) sources.set(name, []);
      sources.get(name).push(source.element);
    });
    const title = document.createElement('strong');
    title.textContent = 'Input matrices';
    const close = document.createElement('button');
    close.type = 'button'; close.textContent = '×'; close.setAttribute('aria-label', 'Close input matrices');
    close.addEventListener('click', closeMathPanel);
    const header = document.createElement('div'); header.append(title, close);
    const list = document.createElement('ul');
    let originalView = null;
    let originalStageSelection = null;
    const back = document.createElement('button');
    back.type = 'button'; back.className = 'input-card-back';
    back.textContent = 'Back to selected cell';
    back.title = 'Back to selected cell';
    back.setAttribute('aria-label', 'Back to selected cell');
    inputReturnCard.replaceChildren(back);
    back.addEventListener('click', () => {
      if(originalStageSelection) {
        window.LittleLlamaWorkbook.restoreStageSelection(originalStageSelection);
        originalStageSelection=null;
      }
      if (originalView && window.LittleLlamaWorkbook.restoreView) window.LittleLlamaWorkbook.restoreView(originalView);
      else if (window.LittleLlamaWorkbook.focusCells) window.LittleLlamaWorkbook.focusCells([cell.element], {preserveZoom: true});
      else zoomToRegion(cell.element, true);
      positionInputCard();
      inputReturnCard.hidden = true;
      returnCardElements = [];
      originalView = null;
    });
    sources.forEach((elements, name) => {
      const item = document.createElement('li');
      const hiddenCard=elements[0]?.closest('.stage-card');
      if(hiddenCard?.style.display==='none') {
        const badge=document.createElement('span');badge.className='hidden-input-badge';
        badge.textContent=`Hidden input: ${hiddenCard.dataset.operation.replaceAll(' • ',' · ')}`;
        item.append(badge);elements.forEach(element=>hiddenInputBadges.set(element,badge));
        const show=document.createElement('button');show.type='button';show.textContent='Show this step';show.className='hidden-input-action';
        show.onclick=()=>{window.LittleLlamaWorkbook.addSelectedStep(hiddenCard.dataset.operation);renderMathPanel(cell);};
        const full=document.createElement('button');full.type='button';full.textContent='View in full model';full.className='hidden-input-action';
        full.onclick=()=>{
          originalView=window.LittleLlamaWorkbook.captureView?.();
          originalStageSelection=window.LittleLlamaWorkbook.stageSelection();
          window.LittleLlamaWorkbook.restoreStageSelection({...originalStageSelection,view:'full'});
          window.LittleLlamaWorkbook.focusCells(elements,{preserveZoom:true});
          inputReturnCard.replaceChildren(back);inputReturnCard.hidden=false;returnCardElements=elements;positionReturnCard();
        };
        item.append(show,full);list.append(item);return;
      }
      const button = document.createElement('button');
      button.className = 'input-matrix-link'; button.type = 'button';
      button.textContent = cell.element.closest('[data-kv-timeline]') ? compactMatrixName(name) : name;
      button.title = name;
      button.addEventListener('click', () => {
        if (inputReturnCard.hidden) originalView = window.LittleLlamaWorkbook.captureView?.();
        returnCardElements = elements;
        // Retain the original output-cell anchor while navigating to inputs.
        // Hide the explanation panel without clearing the selected formula
        // or its dependency connections while navigating to an input.
        mathPanel.classList.remove('open');
        if (window.LittleLlamaWorkbook.focusCells) window.LittleLlamaWorkbook.focusCells(elements, {preserveZoom: true, smooth:true});
        else zoomToRegion(elements[0], true);
        inputReturnCard.hidden = false;
        positionReturnCard();
      });
      item.append(button); list.append(item);
    });
    if (!sources.size) { const item = document.createElement('li'); item.textContent = 'Saved parameter — no input matrices'; list.append(item); }
    inputMatrixCard.replaceChildren(header, list);
    inputMatrixCard.hidden = false;
    positionInputCard();
  }
  window.addEventListener('latte:camera', positionInputCard);
  window.addEventListener('resize', positionInputCard);
  window.addEventListener('scroll', positionInputCard, {passive:true});
  window.addEventListener('latte:camera', positionReturnCard);
  window.addEventListener('resize', positionReturnCard);
  window.addEventListener('scroll', positionReturnCard, {passive:true});
  document.addEventListener('keydown', event => { if (event.key === 'Escape') closeMathPanel(); });

  const dependencyOverlay = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  dependencyOverlay.classList.add("dependency-overlay");
  dependencyOverlay.setAttribute("aria-hidden", "true");
  document.body.appendChild(dependencyOverlay);

  function clearDependencyLinks() {
    dependencyOverlay.replaceChildren();
    document.querySelectorAll(".block-related").forEach((element) => element.classList.remove("block-related"));
  }

  function visibleConnectionTarget(element) {
    const badge=hiddenInputBadges.get(element);
    if(badge?.isConnected)return badge;
    if (!element) return null;
    if (element.offsetParent !== null) return element;
    const row = element.closest("tr");
    const block = blocks.find((candidate) => candidate.rows.includes(row));
    if (block) {
      if (block.group && !block.group.open) {
        const summary = block.group.querySelector('summary');
        summary.classList.add('block-related');
        return summary;
      }
      block.header.classList.add("block-related");
      return block.marker;
    }
    return null;
  }

  function drawDependencyLinks() {
    clearDependencyLinks();
    if (!selectedFormulaCell || selectedFormulaCell.element.offsetParent === null) return;
    const rectFor = element => window.LittleLlamaWorkbook?.screenRect?.(element) || element.getBoundingClientRect();
    const sourceRect = rectFor(selectedFormulaCell.element);
    const source = {
      x: sourceRect.left + window.scrollX + sourceRect.width / 2,
      y: sourceRect.top + window.scrollY + sourceRect.height / 2,
    };
    const targets = [];
    const seen = new Set();
    selectedDependencies.flatMap((dependency) => dependency.elements).forEach((element) => {
      const target = visibleConnectionTarget(element);
      if (!target || target === selectedFormulaCell.element || seen.has(target)) return;
      seen.add(target);
      targets.push(target);
    });
    const width = Math.max(document.documentElement.scrollWidth, document.body.scrollWidth);
    const height = Math.max(document.documentElement.scrollHeight, document.body.scrollHeight);
    dependencyOverlay.setAttribute("width", width);
    dependencyOverlay.setAttribute("height", height);
    dependencyOverlay.setAttribute("viewBox", `0 0 ${width} ${height}`);
    targets.forEach((target) => {
      const rect = rectFor(target);
      const end = {
        x: rect.left + window.scrollX + rect.width / 2,
        y: rect.top + window.scrollY + rect.height / 2,
      };
      const bend = Math.max(30, Math.abs(end.x - source.x) * .35);
      const direction = end.x >= source.x ? 1 : -1;
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.classList.add("dependency-link");
      path.setAttribute("d", `M ${source.x} ${source.y} C ${source.x + bend * direction} ${source.y}, ${end.x - bend * direction} ${end.y}, ${end.x} ${end.y}`);
      const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      dot.classList.add("dependency-link-dot");
      dot.setAttribute("cx", end.x);
      dot.setAttribute("cy", end.y);
      dot.setAttribute("r", 4);
      dependencyOverlay.append(path, dot);
    });
  }

  let linkFrame = 0;
  function scheduleDependencyLinks() {
    cancelAnimationFrame(linkFrame);
    linkFrame = requestAnimationFrame(drawDependencyLinks);
  }

  function closeMathPanel() {
    clearSelectionButton.hidden = true;
    [...selectedDependencies.flatMap(dependency => dependency.elements), ...returnCardElements,
      ...(selectedFormulaCell ? [selectedFormulaCell.element] : [])].forEach(element => {
        changeAnimations.get(element)?.cancel();
        changeAnimations.delete(element);
      });
    inputMatrixCard.hidden = true;
    inputCardCell = null;
    inputReturnCard.hidden = true;
    returnCardElements = [];
    mathPanel.classList.remove("open");
    document.querySelectorAll(".math-selected, .math-dependency, .math-term-selected").forEach((element) =>
      element.classList.remove("math-selected", "math-dependency", "math-term-selected")
    );
    selectedFormulaCell = null;
    selectedDependencies = [];
    clearDependencyLinks();
    document.querySelector('[data-action="focus"]')?.setAttribute('disabled', '');
    restoreFocus();
  }
  mathPanel.querySelector(".math-close").addEventListener("click", closeMathPanel);

  function temperatureSources() {
    return [...cells.values()].filter(source=>
      (source.element.dataset.matrixName==='Top-k candidates' && source.element.style.display!=='none') ||
      (source.element.dataset.matrixName==='Temperature · Top-k · Seed' && source.element.dataset.matrixRow==='1'));
  }

  function displayedLogitSource(cell) {
    return [...cells.values()].filter(source=>source.element.dataset.matrixName==='Logits' && source.element.dataset.matrixRow===cell.element.dataset.matrixRow).sort((a,b)=>b.column-a.column)[0];
  }

  function toggleCellSelection(cell) {
    if(selectedFormulaCell?.element===cell.element){closeMathPanel();return;}
    renderMathPanel(cell);
  }
  function renderMathPanel(cell) {
    selectedFormulaCell = cell;
    document.querySelector('[data-action="focus"]')?.removeAttribute('disabled');
    document.querySelectorAll(".math-selected, .math-dependency").forEach((element) =>
      element.classList.remove("math-selected", "math-dependency")
    );
    cell.element.classList.add("math-selected");
    const dependencies = [];
    collectDependencies(cell.ast, cell, dependencies, new Set());
    if (cell.element.dataset.matrixName === 'Top-k eligibility') {
      dependencies.length=0;
      const source=displayedLogitSource(cell);
      if(source)dependencies.push({label:'Logit at final token position',value:getCellValue(source),elements:[source.element]});
    }
    if (cell.element.dataset.matrixName === 'Top-k candidates') {
      dependencies.length=0;
      const source=[...cells.values()].find(c=>c.element.dataset.matrixName==='Top-k eligibility' && c.element.dataset.matrixRow===cell.element.dataset.matrixRow);
      if(source)dependencies.push({label:'Selected logit',value:source.element.querySelector('font')?.textContent,elements:[source.element]});
    }
    if (cell.element.dataset.matrixName === 'Top-k probabilities') {
      dependencies.length=0;
      temperatureSources().forEach(source=>dependencies.push({label:source.element.dataset.matrixName,value:getCellValue(source),elements:[source.element]}));
    }
    for(let i=dependencies.length-1;i>=0;i--) {
      dependencies[i].elements=dependencies[i].elements.filter(element=>element.style.display!=='none');
      if(!dependencies[i].elements.length)dependencies.splice(i,1);
    }
    selectedDependencies = dependencies;
    dependencies.flatMap((dependency) => dependency.elements).forEach((element) =>
      element.classList.add("math-dependency")
    );
    // Math explainer temporarily disabled pending redesign. Keep selection
    // and dependency connections above, and the input-matrix card below.
    // mathPanel.querySelector(".math-address").textContent = cell.element.closest('.stage-card')?.dataset.operation || cell.sheet;
    // mathPanel.querySelector(".math-result").textContent = `= ${formatValue(cell, getCellValue(cell))}`;
    // mathPanel.querySelector(".math-summary").textContent = formulaExplanation(cell.formula);
    // renderCalculationDetails(cell);
    // mathPanel.classList.add("open");
    showInputMatrices(cell);
    if (focusSnapshot) focusSelected();
    scheduleDependencyLinks();
  }

  formulaCells.forEach((cell) => {
    cell.element.classList.add("calculation-cell");
    cell.element.dataset.cellAddress = `${cell.sheet}!${columnLabel(cell.column)}${cell.row}`;
    cell.element.addEventListener("click", () => toggleCellSelection(cell));
  });

  const blocks = [];
  const layerGroups = [];
  let focusSnapshot = null;
  const forwardTable = document.querySelector('.worksheet[aria-label="Full Forward Pass"] > table');
  const navigationCell = forwardTable ? [...forwardTable.querySelectorAll('td')].find(element =>
    element.textContent.trim() === 'Navigate to block') : null;
  const gridEndColumn = navigationCell ? Number(navigationCell.dataset.gridColumn) - 2 :
    Math.max(2, ...[...cells.values()].filter(cell => cell.sheet === 'Full Forward Pass' &&
      (cell.formula || cell.element.hasAttribute('sdval'))).map(cell => cell.column + cell.element.colSpan - 1));
  if (forwardTable) {
    const rows = [...forwardTable.rows];
    const starts = rows.map((row, index) => {
      const marker = [...row.cells].find((cell) => /^\[\d+\/\d+\]$/.test(cell.textContent.trim()));
      return marker ? { row, marker, index } : null;
    }).filter(Boolean);
    starts.forEach((start, index) => {
      const block = {
        header: start.row,
        marker: start.marker,
        rows: rows.slice(start.index + 1, starts[index + 1]?.index ?? rows.length),
        collapsed: false,
        button: document.createElement("button"),
      };
      block.header.classList.add("block-header");
      block.button.className = "block-toggle";
      block.button.type = "button";
      block.button.textContent = "▾";
      block.button.title = "Collapse block";
      block.button.setAttribute("aria-expanded", "true");
      block.button.addEventListener("click", (event) => {
        event.stopPropagation();
        setBlockCollapsed(block, !block.collapsed);
      });
      start.marker.insertBefore(block.button, start.marker.firstChild);
      blocks.push(block);

    });
  }

  function setBlockCollapsed(block, collapsed) {
    block.collapsed = collapsed;
    block.card?.classList.toggle('is-expanded', !collapsed);
    block.rows.forEach((row) => { row.style.display = block.card ? "" : (collapsed ? "none" : ""); });
    block.button.textContent = collapsed ? "▸" : "▾";
    block.button.title = collapsed ? "Expand block" : "Collapse block";
    block.button.setAttribute("aria-expanded", String(!collapsed));
    block.header.setAttribute('aria-expanded', String(!collapsed));
    scheduleDependencyLinks();
  }

  blocks.forEach((block) => setBlockCollapsed(block, false));

  // Move the original cells into stage cards; their formula objects and event
  // handlers stay attached, so the presentation never duplicates matrix data.
  if (forwardTable && blocks.length) {
    const gridStartColumn = Number(blocks[0].marker.dataset.gridColumn || 2);
    const section = forwardTable.closest('.worksheet');
    const hero = document.createElement('header');
    hero.className = 'flow-hero';
    hero.innerHTML = '<div class="eyebrow">LITTLE LLAMA / MODEL EXPLORER</div><h1>Follow the thought.</h1><p>From characters to predictions. Explore every matrix along the way.</p><div class="flow-legend"><span>01 Embedding</span><i>→</i><span>02 Attention</span><i>→</i><span>03 Feed forward</span><i>→</i><span>04 Prediction</span></div>';
    const flow = document.createElement('div');
    flow.className = 'flow-stages';
    const groups = new Map();
    blocks.forEach((block, index) => {
      const title = block.header.cells[1]?.textContent.trim() || `Stage ${index}`;
      const category = title.split(' • ')[0];
      const card = document.createElement('article');
      card.className = 'stage-card';
      card.dataset.operation = title;
      card.dataset.category = category.toLowerCase();
      const heading = document.createElement('div');
      heading.className = 'stage-heading';
      const number = document.createElement('span');
      number.className = 'stage-number';
      number.textContent = String(index).padStart(2, '0');
      const text = document.createElement('div');
      const tag = document.createElement('span');
      tag.className = 'stage-category';
      tag.textContent = category;
      const name = document.createElement('h2');
      name.textContent = title.split(' • ').slice(1).join(' · ') || title;
      text.append(tag, name);
      heading.append(number, text, block.button);
      const content = document.createElement('div');
      content.className = 'stage-content';
      const table = document.createElement('table');
      table.className = 'matrix-layout';
      const isWatermark = cell => /^LAYER\s+\d+$/i.test(cell.textContent.trim()) && !cell.hasAttribute('sdval');
      // Shared grid origin keeps matrices aligned across every operation.
      const minColumn = gridStartColumn;
      const maxColumn = gridEndColumn;
      table.style.setProperty('--matrix-columns', Math.max(1, maxColumn - minColumn + 1));
      block.rows.forEach(row => {
        [...row.cells].forEach(cell => {
          const column = Number(cell.dataset.gridColumn);
          if (column < minColumn || column > maxColumn || isWatermark(cell)) {
            cell.style.display = 'none';
            return;
          }
          cell.style.gridColumn = `${column - minColumn + 1} / span ${Math.min(cell.colSpan, maxColumn - column + 1)}`;
          if (!cell.textContent.trim() && !cell.hasAttribute('sdval') && !cell.classList.contains('live-input-token')) cell.classList.add('matrix-space');
          if (cell.hasAttribute('sdval')) {
            cell.classList.add('matrix-value');
            cell.style.backgroundColor = cell.getAttribute('bgcolor') || '#edf2f8';
          }
          else if (cell.textContent.trim()) {
            cell.classList.add('matrix-label');
            if (cell.colSpan === 1 && !cell.classList.contains('live-input-token') && cell.getAttribute('align') !== 'center') {
              const next = [...row.cells].find(candidate => Number(candidate.dataset.gridColumn) > column &&
                Number(candidate.dataset.gridColumn) <= maxColumn && !isWatermark(candidate) && candidate.textContent.trim());
              const endColumn = next ? Number(next.dataset.gridColumn) : maxColumn + 1;
              cell.style.gridColumn = `${column - minColumn + 1} / span ${Math.max(1, endColumn - column)}`;
            }
          }
        });
        table.append(row);
        if ([...row.cells].every(cell => cell.style.display === 'none' || cell.classList.contains('matrix-space') ||
          (cell.classList.contains('matrix-label') && !cell.classList.contains('live-input-token')))) {
          row.classList.add('matrix-caption-row');
        }
      });
      content.append(table);
      card.append(heading, content);
      const groupName = title.match(/layer\s+\d+/i)?.[0].replace(/^layer/i, 'Layer') ||
        (category === 'EMBEDDING' ? 'Input embedding' : 'Output & generation');
      if (!groups.has(groupName)) {
        const group = document.createElement('details');
        group.className = 'layer-group';
        group.open = true;
        const summary = document.createElement('summary');
        summary.textContent = groupName;
        const count = document.createElement('span');
        count.className = 'layer-count';
        summary.append(count);
        group.append(summary);
        group.addEventListener('toggle', scheduleDependencyLinks);
        groups.set(groupName, group);
        layerGroups.push(group);
        flow.append(group);
      }
      block.group = groups.get(groupName);
      block.group.append(card);
      const count = block.group.querySelector('.layer-count');
      count.textContent = `${block.group.querySelectorAll('.stage-card').length} operations`;
      block.header = heading;
      block.marker = number;
      block.card = card;
      heading.tabIndex = 0;
      heading.setAttribute('role', 'button');
      heading.setAttribute('aria-label', title);
      // Heading markup is retained for a future return of visible titles.
      /* heading.addEventListener('click', () => setBlockCollapsed(block, !block.collapsed));
      heading.addEventListener('keydown', event => {
        if (event.target === heading && (event.key === 'Enter' || event.key === ' ')) {
          event.preventDefault();
          setBlockCollapsed(block, !block.collapsed);
        }
      }); */
      content.addEventListener('click', event => {
        if (event.target.closest('button')) return;
        if (zoom >= 1) return;
        event.stopPropagation();
        const target = event.target.closest('td.matrix-value') || content;
        zoomToRegion(target);
      }, true);
      content.tabIndex = 0;
      content.setAttribute('role', 'region');
      content.setAttribute('aria-label', `${title} matrix preview`);
      content.addEventListener('keydown', event => {
        if (event.target === content && zoom < 1 && (event.key === 'Enter' || event.key === ' ')) {
          event.preventDefault();
          zoomToRegion(content);
        }
      });
      setBlockCollapsed(block, false);
    });
    forwardTable.style.display = 'none';
    section.append(hero, flow);
    document.body.classList.add('explorer-design');
  }

  function restoreFocus() {
    if (!focusSnapshot) return;
    blocks.forEach((block, index) => {
      block.card.hidden = false;
      setBlockCollapsed(block, focusSnapshot.blocks[index]);
    });
    layerGroups.forEach((group, index) => {
      group.hidden = false;
      group.open = focusSnapshot.groups[index];
    });
    focusSnapshot = null;
    toolbar.querySelector('[data-action="focus"]')?.replaceChildren('Focus selected');
  }

  function focusSelected() {
    if (!selectedFormulaCell) return;
    if (!focusSnapshot) focusSnapshot = {
      blocks: blocks.map(block => block.collapsed),
      groups: layerGroups.map(group => group.open),
    };
    const elements = new Set([selectedFormulaCell.element, ...selectedDependencies.flatMap(dependency => dependency.elements)]);
    blocks.forEach(block => {
      const related = block.rows.some(row => [...row.cells].some(cell => elements.has(cell)));
      block.card.hidden = !related;
      if (related) setBlockCollapsed(block, false);
    });
    layerGroups.forEach(group => {
      group.hidden = !blocks.some(block => block.group === group && !block.card.hidden);
      group.open = !group.hidden;
    });
    toolbar.querySelector('[data-action="focus"]')?.replaceChildren('Restore view');
    scheduleDependencyLinks();
  }

  function setAllBlocks(collapsed) {
    restoreFocus();
    blocks.forEach(block => setBlockCollapsed(block, collapsed));
    layerGroups.forEach(group => { group.open = true; });
  }

  const cursorContext = document.createElement('div');
  const vocabularyLabelStyle = document.createElement('style');
  vocabularyLabelStyle.textContent = '.matrix-value.vocabulary-row-label[data-row-index]::after { left: auto; right: calc(100% + 22px); width: max-content; white-space: pre; } .vocabulary-token-label { position: absolute; right: calc(100% + 5px); top: 50%; transform: translateY(-50%); font: 8px/1.2 ui-sans-serif, system-ui; color: #7c3aed !important; white-space: pre; }';
  document.head.appendChild(vocabularyLabelStyle);
  indexMatrices();
  matrixColorGroups.forEach(({component}) => component.forEach(cell => {
    if (!cell.formula) cell.element.addEventListener('click', () => showInputMatrices(cell));
  }));
  matrixColorGroups.forEach(({component}) => component.forEach(cell => writeCell(cell,getCellValue(cell))));
  refreshMatrixColors();
  function renderSamplingChart() {
    if(!samplingChartBody)return;
    const matrix=name=>[...cells.values()].filter(c=>c.element.dataset.matrixName===name).sort((a,b)=>a.row-b.row);
    const candidates=matrix('Top-k candidates'),p=matrix('Top-k probabilities'),scores=matrix('Sampling score');
    const rows=candidates.map((cell,i)=>({cell,i,logit:getCellValue(cell),p:Number(getCellValue(p[i])),score:Number(getCellValue(scores[i]))})).filter(row=>typeof row.logit==='number'&&Number.isFinite(row.logit));
    samplingChartBody.replaceChildren();
    if(!rows.length||rows.some(row=>!Number.isFinite(row.p)||!Number.isFinite(row.score))) {samplingChartBody.textContent='Enter valid tokens to view sampling.';return;}
    const max=Math.max(...rows.map(row=>row.logit)),weights=rows.map(row=>Math.exp(row.logit-max)),sum=weights.reduce((a,b)=>a+b,0),scoreSum=rows.reduce((total,row)=>total+row.score,0);
    const best=Math.max(...rows.map(row=>row.score));
    const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg'),height=rows.length*52+48;
    svg.setAttribute('viewBox',`0 0 430 ${height}`);svg.setAttribute('role','img');svg.setAttribute('aria-label','Grouped horizontal bars comparing top-k, temperature probabilities, and normalized randomized scores on a zero to 100 percent scale');
    const draw=(tag,attrs,text)=>{const node=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>node.setAttribute(k,v));if(text!==undefined)node.textContent=text;svg.append(node);return node;};
    const x=48,width=315;
    [0,.25,.5,.75,1].forEach(value=>{
      draw('line',{x1:x+value*width,x2:x+value*width,y1:22,y2:height-12,stroke:'#e0e6ef'});
      draw('text',{x:x+value*width,y:13,'text-anchor':'middle',fill:'#637791','font-size':10},`${value*100}%`);
    });
    rows.forEach((row,j)=>{
      const top=26+j*52,selected=row.score===best;
      if(selected)draw('rect',{x:0,y:top-4,width:430,height:47,fill:'#f3edfc',rx:5});
      const token=cells.get(key('Embeddings',row.i+6,1));
      draw('text',{x:x-9,y:top+20,'text-anchor':'end',fill:'#7c3aed','font-size':12},displayToken(token?getCellValue(token):''));
      const series=[weights[j]/sum,row.p,scoreSum>0?row.score/scoreSum:0],colors=['#287baa','#dd8a32','#9565db'],names=['Top-k at T = 1','Temperature probability','Randomized normalized score'];
      series.forEach((value,k)=>{
        const y=top+k*12;
        const bar=draw('rect',{x,y,width:Math.max(0,Math.min(1,value))*width,height:8,fill:colors[k]});
        const title=document.createElementNS(ns,'title');title.textContent=`${names[k]}: ${(value*100).toFixed(2)}%`;bar.append(title);
        draw('text',{x:x+value*width+4,y:y+8,fill:'#526780','font-size':8},`${(value*100).toFixed(1)}%`);
      });
      if(selected)draw('text',{x:423,y:top+40,'text-anchor':'end',fill:'#7c3aed','font-size':9},'Selected');
    });
    samplingChartBody.append(svg);
  }
  function refreshSamplingLayout() {
    const matrix=name=>[...cells.values()].filter(c=>c.element.dataset.matrixName===name).sort((a,b)=>a.row-b.row);
    const eligibility=matrix('Top-k eligibility');
    if(!eligibility.length)return;
    const rows=eligibility.map(c=>c.element.closest('tr'));
    const retained=eligibility.map(c=>getCellValue(c)===1);
    eligibility.forEach((cell,i)=>{
      let mark=cell.element.querySelector('.sampling-mark');
      if(!mark){mark=document.createElement('span');mark.className='sampling-mark';cell.element.append(mark);}
      mark.textContent=retained[i]?'✓':'×';
      mark.style.color=retained[i]?'#27835d':'#8995a6';
    });
    const scores=matrix('Sampling score');
    const best=Math.max(...scores.map(getCellValue));
    const winner=scores.find(cell=>getCellValue(cell)===best);
    scores.forEach(cell=>cell.element.querySelector('.sampling-selected-card')?.remove());
    for(const name of ['Top-k candidates','Top-k probabilities','Sampling noise · saved seeded draw','Sampling score']) {
      let compact=0;
      matrix(name).forEach((cell,i)=>{
        delete cell.element.dataset.columnIndex;
        cell.element.style.setProperty('display',retained[i]?'flex':'none','important');
        if(retained[i]) {
          // Single-column sampling vectors need no column index.
          rows[compact++].appendChild(cell.element);
        }
        if(!cell.element.querySelector('.vocabulary-token-label')) {
          const source=cells.get(key('Embeddings',i+6,1));
          const token=source?getCellValue(source):'';
          const span=document.createElement('span');span.className='vocabulary-token-label';
          span.textContent=token===' '||token==='<space>'?'␣':token==='\n'?'\\n':token==='\t'?'\\t':token;
          cell.element.append(span);
        }
        if(name==='Sampling score')cell.element.style.minWidth='44px';
      });
    }
    const probabilities=matrix('Top-k probabilities');
    renderSamplingChart();
    if(winner){
      const token=cells.get(key('Embeddings',Number(winner.element.dataset.matrixRow)+5,1));
      const card=document.createElement('span');card.className='sampling-selected-card';
      card.textContent=`Selected: ${displayToken(token?getCellValue(token):'')}`;
      winner.element.append(card);
    }
    window.dispatchEvent(new Event('latte:layout'));
  }
  const samplingAnnotationStyle=document.createElement('style');
  samplingAnnotationStyle.textContent='.sampling-mark { position:absolute; left:calc(100% + 5px); top:50%; transform:translateY(-50%); font:10px/1 system-ui; } .sampling-selected-card { position:absolute; left:calc(100% + 12px); top:50%; transform:translateY(-50%); white-space:pre; padding:5px 8px; border:1px solid #d8c5ed; border-radius:6px; background:#f5effc; color:#7c3aed !important; font:10px/1.3 system-ui; } body.color-overview .sampling-mark, body.color-overview .sampling-selected-card { visibility:hidden; }';
  document.head.append(samplingAnnotationStyle);
  // Present the sampling pipeline in its existing matrix section. Noise stays
  // in the formula graph; the fourth column displays p / E and the winner.
  document.querySelectorAll('.stage-card').forEach(card=>{
    if(!card.dataset.operation.includes('Seeded top-k sampling'))return;
    card.querySelectorAll('.matrix-label').forEach(label=>{
      const names={'Top-k eligibility':'Logits · ✓ / ×','Top-k candidates':'Top-k only','Top-k probabilities':'Temperature','Sampling score':'Randomized · p / E'};
      const text=label.textContent.trim();
      if(names[text])label.textContent=names[text];
    });
  });
  refreshSamplingLayout();
  document.querySelectorAll('.stage-card').forEach(card=>{
    if(card.dataset.operation==='OUTPUT • Vocabulary softmax' && !document.documentElement.hasAttribute('data-training-replay'))card.style.display='none';
  });
  cursorContext.className = 'cursor-context';
  cursorContext.setAttribute('aria-hidden', 'true');
  document.body.append(cursorContext);
  document.addEventListener('pointermove', event => {
    const card = event.target.closest?.('.stage-card');
    if (!card || event.pointerType === 'touch') {
      cursorContext.classList.remove('visible');
      return;
    }
    const cell = event.target.closest('td');
    const operation = card.dataset.operation || '';
    const layer = operation.match(/layer\s+\d+/i)?.[0];
    const name = operation.replace(/\s*•\s*layer\s+\d+/i, '').replaceAll(' • ', ' · ');
    const address = cell?.dataset.cellAddress?.split('!').pop();
    cursorContext.textContent = [layer, name, address].filter(Boolean).join(' / ');
    cursorContext.classList.add('visible');
    const bounds = cursorContext.getBoundingClientRect();
    cursorContext.style.left = `${Math.max(8, Math.min(event.clientX + 16, window.innerWidth - bounds.width - 8))}px`;
    cursorContext.style.top = `${Math.max(8, Math.min(event.clientY + 18, window.innerHeight - bounds.height - 8))}px`;
  });
  document.addEventListener('pointerleave', () => cursorContext.classList.remove('visible'));
  window.addEventListener('scroll', () => cursorContext.classList.remove('visible'), { passive: true });

  const controlRail = document.createElement("aside");
  controlRail.className = "explorer-control-rail";
  controlRail.setAttribute("aria-label", "Visualization controls");
  document.body.appendChild(controlRail);

  const sequenceEditor = document.createElement("div");
  sequenceEditor.className = "sequence-editor";
  sequenceEditor.setAttribute("aria-label", "Input sequence editor");
  const sequenceTitle = document.createElement("span");
  sequenceTitle.className = "sequence-title";
  sequenceTitle.textContent = "Input sequence";
  sequenceEditor.appendChild(sequenceTitle);
  const sequenceInputs = [];
  [...editableCells.values()].sort((a, b) => a.column - b.column).forEach((cell, index) => {
    const input = document.createElement("input");
    input.className = "sequence-token";
    input.value = displayToken(cell.baseValue);
    cell.element.textContent = displayToken(cell.baseValue);
    input.setAttribute("aria-label", `Sequence token ${index}`);
    input.addEventListener("input", () => {
      const target = cell.element.querySelector("font") || cell.element;
      target.textContent = inputToken(input.value);
      cell.element.dispatchEvent(new Event("input", { bubbles: true }));
    });
    cell.element.addEventListener("input", () => {
      input.value = displayToken(inputToken(cell.element.textContent.replace(/[\r\n]+/g, "")));
    });
    input.addEventListener("focus", () => input.select());
    input.addEventListener("click", () => input.select());
    input.addEventListener("keydown", event => {
      if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
      const step = event.key === "ArrowLeft" || event.key === "ArrowUp" ? -1
        : event.key === "ArrowRight" || event.key === "ArrowDown" ? 1 : 0;
      const targetIndex = event.key === "Home" ? 0
        : event.key === "End" ? sequenceInputs.length - 1
        : index + step;
      if ((!step && event.key !== "Home" && event.key !== "End") || !sequenceInputs[targetIndex]) return;
      event.preventDefault();
      sequenceInputs[targetIndex].focus();
      sequenceInputs[targetIndex].select();
    });
    sequenceInputs.push(input);
    sequenceEditor.appendChild(input);
  });
  controlRail.appendChild(sequenceEditor);
  const generationControls = JSON.parse(document.getElementById('generation-controls')?.textContent || '{}');
  const generationSettings = document.createElement('div');
  generationSettings.className = 'generation-settings';
  generationSettings.setAttribute('aria-label','Sampling settings');
  Object.entries(generationControls).forEach(([name,address]) => {
    const match = address.match(/^R(\d+)C(\d+)$/);
    if (!match) return;
    const cell = cells.get(key('Full Forward Pass',Number(match[1]),Number(match[2])));
    if (!cell) return;
    const label = document.createElement('label');
    label.className = 'generation-control';
    label.textContent = {temperature:'Temperature',top_k:'Top-k',seed:'Seed'}[name];
    const input = document.createElement('input');
    input.type = 'number'; input.value = cell.baseValue;
    input.step = name === 'temperature' ? '.1' : '1';
    if (name !== 'seed') input.min = '0';
    input.addEventListener('change',() => {
      const value = Number(input.value);
      if (!Number.isFinite(value) || (name !== 'seed' && value < 0) || (name !== 'temperature' && !Number.isInteger(value))) {
        input.value=cell.baseValue; return;
      }
      cell.baseValue=value;
      writeCell(cell,value);
      recalculate();
    });
    label.append(input); generationSettings.append(label);
  });
  if (generationSettings.childElementCount) controlRail.append(generationSettings);
  if(generationSettings.childElementCount) {
    const chartToggle=document.createElement('button');chartToggle.type='button';chartToggle.className='toolbar-button sampling-chart-toggle';
    chartToggle.title='Show or hide sampling chart';chartToggle.setAttribute('aria-label','Show or hide sampling chart');chartToggle.setAttribute('aria-controls','sampling-chart-card');
    chartToggle.innerHTML='<svg width="18" height="18" viewBox="0 0 20 20" aria-hidden="true"><path d="M3 2v15h15M6 13V9m4 4V5m4 8V3" fill="none" stroke="currentColor" stroke-width="2"/></svg>';
    generationSettings.append(chartToggle);
    const card=document.createElement('section');card.id='sampling-chart-card';card.className='sampling-chart-card';card.hidden=true;
    card.setAttribute('aria-label','Sampling comparison chart');
    card.innerHTML='<header><strong>Sampling comparison</strong><button type="button" aria-label="Close sampling chart" title="Close sampling chart">×</button></header><div class="sampling-chart-legend"><span style="color:#287baa">● Top-k</span><span style="color:#dd8a32">● Temperature</span><span style="color:#9565db">● Randomized</span></div><p>Top-k uses T = 1. Randomized shows normalized p / E scores, not selection probabilities. Each series sums to 100%.</p><div class="sampling-chart-body"></div>';
    document.body.append(card);samplingChartBody=card.querySelector('.sampling-chart-body');
    setSamplingChartOpen=open=>{open=open&&!chartToggle.disabled;card.hidden=!open;chartToggle.setAttribute('aria-expanded',String(open));chartToggle.classList.toggle('active',open);if(open)renderSamplingChart();};
    setSamplingChartOpen(false);
    chartToggle.onclick=()=>setSamplingChartOpen(card.hidden);card.querySelector('header button').onclick=()=>setSamplingChartOpen(false);
    const style=document.createElement('style');
    style.textContent='.sampling-chart-card { position:fixed; z-index:130; right:18px; bottom:112px; width:min(480px,calc(100vw - 36px)); max-height:calc(100vh - 220px); overflow:auto; box-sizing:border-box; padding:16px; border:1px solid #dce4ef; border-radius:12px; background:#fff; box-shadow:0 8px 35px #23304724; font:12px/1.4 system-ui; color:#405978; } .sampling-chart-card { overscroll-behavior:contain; touch-action:pan-y; } .sampling-chart-card[hidden] { display:none; } .sampling-chart-card header { display:flex; align-items:center; justify-content:space-between; } .sampling-chart-card header button { border:0; background:transparent; font-size:20px; cursor:pointer; color:#70819a; } .sampling-chart-legend { display:flex; gap:16px; margin:12px 0 6px; } .sampling-chart-card p { color:#6c7d92; font-size:11px; } .sampling-chart-body svg { display:block; width:100%; height:auto; } .sampling-chart-toggle.active { background:#eee6fa; color:#7c3aed; }';
    document.head.append(style);
  }

  const selectedTokenCell = [...cells.values()].find(cell =>
    cell.element.dataset.matrixName === 'Selected vocabulary index · 1-based');
  if (selectedTokenCell) {
    const topToken = document.createElement('span');
    topToken.className = 'generated-token-output';
    topToken.setAttribute('aria-live','polite');
    sequenceEditor.appendChild(topToken);
    const bottomToken = document.createElement('div');
    bottomToken.className = 'generated-token-output generated-token-bottom';
    bottomToken.style.marginLeft = `${(parseInt(selectedTokenCell.element.style.gridColumn,10))*44}px`;
    bottomToken.style.transform = 'translateX(-100%)';
    bottomToken.style.marginRight = '0';
    bottomToken.setAttribute('aria-live','polite');
    selectedTokenCell.element.closest('.stage-content').appendChild(bottomToken);
    const updateGeneratedToken = () => {
      const index = getCellValue(selectedTokenCell);
      const source = Number.isInteger(index) && index > 0 ? cells.get(key('Embeddings',index+5,1)) : null;
      const token = source ? getCellValue(source) : null;
      const display = token === ' ' || token === '<space>' ? '␣' : token === '\n' ? '<newline>' : token === '\t' ? '<tab>' : token;
      topToken.textContent = bottomToken.textContent = `→ Generated: ${display ?? 'unavailable'}`;
    };
    document.addEventListener('latte:recalculated',updateGeneratedToken);
    updateGeneratedToken();
  }

  const toolbar = document.createElement("div");
  toolbar.className = "workbook-toolbar";
  toolbar.setAttribute("aria-label", "Workbook controls");
  toolbar.innerHTML = `
    <div class="toolbar-group"><button class="toolbar-button" data-action="sequence" type="button" title="Show or hide input sequence">Sequence</button></div>
    <div class="toolbar-divider"></div>
    <div class="toolbar-group"><button class="toolbar-button" data-action="zoom-out" type="button" title="Zoom out">−</button><span class="zoom-label">25%</span><button class="toolbar-button" data-action="zoom-in" type="button" title="Zoom in">+</button><button class="toolbar-button" data-action="zoom-reset" type="button" title="Return to overview">Overview</button></div>`;
  controlRail.appendChild(toolbar);

  let zoom = 1;
  function zoomToRegion(target, preserveZoom = false) {
    if (!preserveZoom) setZoom(1);
    requestAnimationFrame(() => {
      const rect = target.getBoundingClientRect();
      window.scrollTo({
        left: Math.max(0, window.scrollX + rect.left + rect.width / 2 - window.innerWidth / 2),
        top: Math.max(0, window.scrollY + rect.top + rect.height / 2 - window.innerHeight / 2),
        behavior: 'smooth',
      });
    });
  }
  function setZoom(nextZoom) {
    zoom = Math.max(0.15, Math.min(2, Math.round(nextZoom * 20) / 20));
    document.documentElement.style.setProperty("--workbook-zoom", zoom);
    const colorOnly = zoom <= 0.3;
    document.body.classList.toggle("color-overview", colorOnly);
    toolbar.querySelector(".zoom-label").textContent = `${Math.round(zoom * 100)}%${colorOnly ? " · colors" : ""}`;
    scheduleDependencyLinks();
  }
  toolbar.addEventListener("click", (event) => {
    const action = event.target.closest("[data-action]")?.dataset.action;
    if (action === "zoom-in") setZoom(zoom + 0.1);
    if (action === "zoom-out") setZoom(zoom - 0.1);
    if (action === "zoom-reset") { closeMathPanel(); setZoom(.25); }
    if (action === "collapse") setAllBlocks(true);
    if (action === "expand") setAllBlocks(false);
    if (action === "focus") { if (focusSnapshot) restoreFocus(); else focusSelected(); }
    if (action === "sequence") sequenceEditor.classList.toggle("hidden");
  });
  setZoom(.25);
  window.addEventListener("resize", scheduleDependencyLinks);
  window.addEventListener('latte:camera', () => {
    cancelAnimationFrame(linkFrame);
    if (selectedFormulaCell) drawDependencyLinks();
  });
  window.addEventListener("scroll", scheduleDependencyLinks, { passive: true });

  function recordedLossSource(cell){
    const row=Number(cell.element.dataset.recordedTargetRow);
    if(!row)return null;
    return [...cells.values()].find(source=>source.column===cell.column &&
      source.element.dataset.matrixName==='Vocabulary probabilities' &&
      Number(source.element.dataset.matrixRow)===row);
  }

  window.LittleLlamaWorkbook = {
    presentationCell(element) {
      const context = [...cells.values()].find(cell => cell.element === element);
      if (!context) return null;
      const sources = new Map();
      const source = address => {
        const id = key(address.sheet, address.row, address.column);
        const cell = cells.get(id);
        if (cell && !sources.has(id)) sources.set(id, {
          id, element:cell.element, name:cell.element.dataset.matrixName || cell.sheet,
          row:Number(cell.element.dataset.matrixRow)-1,
          column:Number(cell.element.dataset.matrixColumn)-1,
        });
        return id;
      };
      function describe(node) {
        if (!node) return null;
        if (node.type === 'reference') return {type:'reference', id:source(resolveReference(node, context))};
        if (node.type === 'range') {
          const first = resolveReference(node.start, context);
          const last = resolveReference(node.end, context, first.sheet);
          const ids=[];
          for(let r=Math.min(first.row,last.row);r<=Math.max(first.row,last.row);r++)
            for(let c=Math.min(first.column,last.column);c<=Math.max(first.column,last.column);c++)
              ids.push(source({sheet:first.sheet,row:r,column:c}));
          return {type:'range',ids};
        }
        if(node.type==='call'&&node.name==='IF')return describe(node.args[evaluate(node.args[0],context)?1:2]);
        if(node.type==='binary')return {...node,left:describe(node.left),right:describe(node.right)};
        if(node.type==='unary')return {...node,value:describe(node.value)};
        if(node.type==='call')return {...node,args:node.args.map(describe)};
        return {...node};
      }
      // Character predictions are nested IF lookups around MATCH(MAX(p),p,0).
      // Trace the selector, rather than the literal character branch it picks.
      const selector=element.dataset.matrixName==='Predicted next character' &&
        context.ast?.name==='IF' ? context.ast.args[0]?.left : null;
      const probability=recordedLossSource(context);
      const node=probability?{type:'unary',operator:'-',value:{type:'call',name:'LN',args:[{
        type:'reference',id:source({sheet:probability.sheet,row:probability.row,column:probability.column}),
      }]}}:describe(selector?.name==='MATCH'?selector:context.ast);
      return {node,sources:[...sources.values()],name:element.dataset.matrixName || '',
        value:getCellValue(context),
        recordedLoss:probability?{probability:Math.exp(-Number(getCellValue(context))),updated:document.documentElement.hasAttribute('data-training-updated')}:null,
        weight:element.classList.contains('learned-parameter'),
        row:Number(element.dataset.matrixRow)-1,column:Number(element.dataset.matrixColumn)-1};
    },
    clearPresentationSelection() { closeMathPanel(); },
    presentationMatrix(elements) {
      const selected=new Set(elements),dependencies=[],seen=new Set();
      for(const cell of cells.values()) {
        if(selected.has(cell.element)){
          collectDependencies(cell.ast,cell,dependencies,seen);
          const probability=recordedLossSource(cell);
          if(probability)dependencies.push({elements:[probability.element]});
        }
      }
      const groups=new Map();
      for(const dependency of dependencies)for(const element of dependency.elements){
        const name=element.dataset.matrixName;
        if(!name||selected.has(element))continue;
        if(!groups.has(name))groups.set(name,new Set());
        groups.get(name).add(element);
      }
      return [...groups].map(([name,items])=>({name,elements:[...items]}));
    },
    setSamplingChartOpen(open) { setSamplingChartOpen(open); },
    setSamplingNoise(draws) {
      const noiseCells=[...cells.values()].filter(cell=>cell.element.dataset.matrixName==='Sampling noise · saved seeded draw').sort((a,b)=>a.row-b.row);
      noiseCells.forEach((cell,i)=>{
        cell.baseValue=draws[i];cell.formula='='+draws[i];
        cell.ast=new FormulaParser(cell.formula).parse();
        cell.element.setAttribute('data-sheets-formula',cell.formula);
      });
    },
    updateTrainingGrid(gridCells) {
      // The replay has a fixed architecture and sequence length: retain the
      // DOM, parsed formulas, annotation layout, and camera between batches.
      deferTrainingPaint=document.documentElement.hasAttribute('data-training-replay');
      const updates = gridCells.map(update => ({update, cell: cells.get(key('Full Forward Pass', update.r, update.c))}));
      if (updates.some(({cell}) => !cell)) throw new Error('Training grid shape changed');
      updates.forEach(({update, cell}) => {
        cell.baseValue = update.v;
        if(update.recordedTargetRow)cell.element.dataset.recordedTargetRow=update.recordedTargetRow;
        if (cell.formula !== update.f) {
          if (!!cell.formula !== !!update.f) throw new Error('Training formula topology changed');
          cell.formula = update.f;
          cell.parseError = null;
          cell.ast = update.f ? new FormulaParser(update.f).parse() : null;
          if (update.f) cell.element.setAttribute('data-sheets-formula', update.f);
        }
        if (!cell.formula) writeCell(cell, cell.baseValue);
      });
      recalculate(false);
    },
    commitSequenceInputs() {
      [...editableCells.values()].sort((a,b) => a.column-b.column).forEach((cell,index) => {
        cell.baseValue = inputToken(sequenceInputs[index].value);
      });
    },
    recalculate,
    erf,
    evaluateGrid(grid) {
      // Calculate a growing sequence without rebuilding any document or UI.
      const saved=new Map(cells),result=new Map();
      try {
        cells.clear();
        saved.forEach((cell,address)=>{if(cell.sheet!=='Full Forward Pass')cells.set(address,cell);});
        for(const c of grid.cells.values()) {
          const cell={sheet:'Full Forward Pass',row:c.r,column:c.c,baseValue:c.v,formula:c.f,
            ast:c.f?new FormulaParser(c.f).parse():null,version:-1,evaluating:false};
          cells.set(key(cell.sheet,c.r,c.c),cell);
        }
        calculationVersion++;
        for(const c of grid.cells.values()) {
          const value=getCellValue(cells.get(key('Full Forward Pass',c.r,c.c)));
          if(isError(value))throw Error(`Calculation failed at R${c.r}C${c.c}: ${value.message}`);
          result.set(`${c.r},${c.c}`,value);
        }
        return result;
      } finally {
        cells.clear();saved.forEach((cell,address)=>cells.set(address,cell));calculationVersion++;
      }
    },
    updateCacheGrid(grid,values,{top,operation}) {
      const card=blocks.find(block=>block.card?.dataset.operation===operation)?.card;
      if(!card)throw Error('Cache stage is unavailable');
      const block=blocks.find(block=>block.card===card),table=card.querySelector('.matrix-layout');
      const identity=(name,r,c)=>`${name}|${r}|${c}`;
      const previous=new Map(),oldRects=new Map();
      card.querySelectorAll('.matrix-value').forEach(element=>{
        const cell=cells.get(key('Full Forward Pass',Number(element.dataset.cellAddress.match(/!([A-Z]+)(\d+)$/)[2]),
          Number(element.dataset.gridColumn)));
        if(cell)previous.set(identity(element.dataset.matrixName,element.dataset.matrixRow,element.dataset.matrixColumn),cell);
        oldRects.set(element,element.getBoundingClientRect());
      });
      const metadata=new Map();
      grid.matrixBlocks.forEach(matrix=>{
        for(let r=0;r<matrix.rows;r++)for(let c=0;c<matrix.cols;c++)metadata.set(`${matrix.row+r},${matrix.column+c}`,{matrix,r:r+1,c:c+1});
      });
      [...cells].forEach(([address,cell])=>{if(cell.sheet==='Full Forward Pass')cells.delete(address);});
      for(let i=formulaCells.length-1;i>=0;i--)if(formulaCells[i].sheet==='Full Forward Pass')formulaCells.splice(i,1);
      matrixColorGroups.splice(0,matrixColorGroups.length);
      gridNames.clear();calculationVersion++;
      const rows=new Map(),components=new Map(),newElements=[];
      for(const c of grid.cells.values()) {
        const info=metadata.get(`${c.r},${c.c}`),visible=c.r>=top&&c.r<grid.row;
        const existing=visible&&info?previous.get(identity(info.matrix.name,info.r,info.c)):null;
        const element=existing?.element||document.createElement('td');
        const cell=existing||{sheet:'Full Forward Pass',element};
        Object.assign(cell,{row:c.r,column:c.c,baseValue:values.get(`${c.r},${c.c}`),formula:c.f,
          ast:visible&&c.f?new FormulaParser(c.f).parse():null,version:calculationVersion,
          cachedValue:values.get(`${c.r},${c.c}`),evaluating:false});
        cells.set(key(cell.sheet,c.r,c.c),cell);
        element.dataset.gridColumn=c.c;
        if(info) {
          element.dataset.matrixName=info.matrix.name;element.dataset.matrixRow=info.r;element.dataset.matrixColumn=info.c;
          gridNames.set(key(cell.sheet,c.r,c.c),`${info.matrix.name}[${info.r-1},${info.c-1}]`);
          if(typeof c.v==='number')element.setAttribute('sdval',cell.baseValue);
        }
        if(!visible)continue;
        if(!rows.has(c.r))rows.set(c.r,[]);
        rows.get(c.r).push(element);
        element.style.gridColumn=`${c.c} / span 1`;
        if(typeof c.v==='number') {
          element.classList.add('matrix-value','calculation-cell');
          element.dataset.cellAddress=`Full Forward Pass!${columnLabel(c.c)}${c.r}`;
          element.setAttribute('data-sheets-formula',c.f||'');
          if(c.f)formulaCells.push(cell);
          if(info.r===1&&info.matrix.cols>1)element.dataset.columnIndex=info.c-1;else delete element.dataset.columnIndex;
          if(info.c===1&&info.matrix.rows>1)element.dataset.rowIndex=info.r-1;else delete element.dataset.rowIndex;
          if(!existing)element.addEventListener('click',()=>toggleCellSelection(cell));
          if(!components.has(info.matrix.name))components.set(info.matrix.name,[]);
          components.get(info.matrix.name).push(cell);
          writeCell(cell,cell.baseValue);
          if(!existing)newElements.push(element);
        } else {
          element.className=c.v?'matrix-label':'matrix-space';element.textContent=c.v;
        }
      }
      const fragment=document.createDocumentFragment();
      for(let r=top;r<grid.row;r++) {
        const row=document.createElement('tr'),elements=rows.get(r)||[];
        if(!elements.some(element=>element.classList.contains('matrix-value')))row.classList.add('matrix-caption-row');
        if(elements.some(element=>element.textContent==='KV attention spacing'))row.classList.add('kv-section-gap');
        if(elements.some(element=>/^(Prefill ·|Decode \d+ · token)/.test(element.textContent)))row.classList.add('kv-step-titles');
        elements.sort((a,b)=>Number(a.dataset.gridColumn)-Number(b.dataset.gridColumn)).forEach(element=>row.append(element));
        fragment.append(row);
      }
      table.style.setProperty('--matrix-columns',grid.width+1);table.replaceChildren(fragment);
      block.rows=[...table.rows];
      components.forEach(component=>matrixColorGroups.push({component,weight:false}));
      if(card.dataset.kvTimeline!=='true')refreshMatrixColors();
      const animateUpdate=()=>{
      if(!matchMedia('(prefers-reduced-motion: reduce)').matches) {
        const moves=[];
        oldRects.forEach((before,element)=>{
          if(!element.isConnected)return;
          const after=element.getBoundingClientRect(),dx=before.left-after.left,dy=before.top-after.top;
          if(Math.abs(dx)+Math.abs(dy)>.5)moves.push({element,dx,dy});
        });
        moves.forEach(({element,dx,dy})=>element.animate([{transform:`translate(${dx}px,${dy}px)`},{transform:'translate(0,0)'}],{duration:180,easing:'ease-out'}));
        newElements.forEach(element=>element.animate([{opacity:0},{opacity:1}],{duration:180,easing:'ease-out'}));
      }
      };
      if(selectedFormulaCell&&selectedFormulaCell.element.isConnected)renderMathPanel(selectedFormulaCell);
      else closeMathPanel();
      scheduleDependencyLinks();
      return animateUpdate;
    },
    setZoom,
    collapseAll() { setAllBlocks(false); setZoom(.25); },
    expandAll() { setAllBlocks(false); },
    focusSelected,
    formulaCount: formulaCells.length,
    editableCellCount: editableCells.size,
    value(sheet, row, column) {
      return getCellValue(cells.get(key(sheet, row, column)));
    },
    matrixData() {
      return [...cells.values()].filter(cell=>cell.element.dataset.matrixName).map(cell=>({
        name:cell.element.dataset.matrixName,
        row:Number(cell.element.dataset.matrixRow)-1,
        column:Number(cell.element.dataset.matrixColumn)-1,
        value:getCellValue(cell),element:cell.element,
      }));
    },
    compactMatrixName,
    refreshColors:refreshMatrixColors,
  };
  const stageCards=[...document.querySelectorAll('.stage-card')];
  const originalDisplays=new Map(stageCards.map(card=>[card,card.style.display]));
  let chosenStages=[],stageView='full';
  const viewTabs=document.createElement('nav');viewTabs.className='stage-view-tabs';viewTabs.setAttribute('aria-label','Visualization views');
  const fullTab=document.createElement('button');fullTab.textContent='Full model';fullTab.type='button';
  const selectedTab=document.createElement('button');selectedTab.textContent='Selected steps';selectedTab.type='button';
  const corpusTab=document.createElement('button');corpusTab.textContent='Training corpus';corpusTab.type='button';
  const corpusPanel=document.createElement('section');corpusPanel.className='training-corpus-panel';corpusPanel.hidden=true;corpusPanel.setAttribute('aria-label','Training corpus');
  const corpusTitle=document.createElement('h2');corpusTitle.textContent='Training corpus';
  const corpusSource=document.createElement('p'),corpusText=document.createElement('pre');
  corpusPanel.append(corpusTitle,corpusSource,corpusText);document.body.append(corpusPanel);
  const chooseButton=document.createElement('button');chooseButton.textContent='Choose from list…';chooseButton.type='button';
  const visualButton=document.createElement('button');visualButton.textContent='Pick on diagram';visualButton.type='button';visualButton.setAttribute('aria-pressed','false');
  const clearStagesButton=document.createElement('button');clearStagesButton.textContent='Reset selected steps';clearStagesButton.type='button';clearStagesButton.title='Deselect all steps and return to Full model';
  const stepsMenu=document.createElement('details');stepsMenu.className='steps-menu';
  const stepsSummary=document.createElement('summary');stepsSummary.textContent='Choose steps';
  const stepsActions=document.createElement('div');stepsActions.className='steps-menu-actions';stepsActions.append(visualButton,chooseButton,clearStagesButton);
  stepsMenu.append(stepsSummary,stepsActions);
  viewTabs.append(fullTab,selectedTab,stepsMenu);document.body.append(viewTabs);
  [visualButton,chooseButton,clearStagesButton].forEach(button=>button.addEventListener('click',()=>{stepsMenu.open=false;}));
  const diagramChecks=new Map();
  stageCards.forEach(card=>{
    const control=document.createElement('label');control.className='stage-pick-control';control.hidden=true;
    const checkbox=document.createElement('input');checkbox.type='checkbox';checkbox.setAttribute('aria-label',`Select ${card.dataset.operation}`);
    control.title=`Select ${card.dataset.operation}`;control.append(checkbox);card.append(control);
    control.addEventListener('click',event=>event.stopPropagation());
    control.addEventListener('pointerdown',event=>event.stopPropagation());
    diagramChecks.set(card.dataset.operation,checkbox);
    checkbox.onchange=()=>{
      chosenStages=stageCards.filter(item=>diagramChecks.get(item.dataset.operation).checked).map(item=>item.dataset.operation);
      list.querySelectorAll('input[data-stage-leaf]').forEach(input=>input.checked=chosenStages.includes(input.value));
      selectedTab.disabled=!chosenStages.length;updateApply();
      card.classList.toggle('stage-picked',checkbox.checked);
      window.dispatchEvent(new Event('latte:stage-selection'));
    };
  });
  let visualSelecting=false;
  const sizeStageSelectors=()=>{
    // These controls are hidden outside diagram picking. Avoid scanning the
    // entire model for each stage on every camera/annotation update.
    if(!visualSelecting)return;
    const scale=window.LittleLlamaWorkbook.cameraZoom?.()||Number(document.documentElement.style.getPropertyValue('--workbook-zoom'))||.25;
    const labels=new Map([...document.querySelectorAll('.operation-annotation[data-operation-index]')]
      .filter(label=>label.style.display!=='none')
      .map(label=>[Number(label.dataset.operationIndex),label]));
    stageCards.forEach((card,index)=>{
      const control=diagramChecks.get(card.dataset.operation).parentElement;
      const label=labels.get(index);
      if(!label){control.style.visibility='hidden';return;}
      control.style.transform=`scale(${1/scale})`;
      control.style.transformOrigin='top left';
      const cardRect=card.getBoundingClientRect();
      if(label) {
        const rect=label.getBoundingClientRect();
        control.style.left=`${(rect.right-cardRect.left+8)/scale}px`;
        control.style.top=`${((rect.top+rect.bottom)/2-cardRect.top-6.5)/scale}px`;
        control.style.right='auto';
        control.style.visibility='visible';
      } else {
        control.style.visibility='hidden';
      }
    });
  };
  window.addEventListener('latte:camera',sizeStageSelectors);
  window.addEventListener('resize',sizeStageSelectors);
  window.addEventListener('latte:annotations',sizeStageSelectors);
  visualButton.onclick=()=>{
    visualSelecting=!visualSelecting;
    if(visualSelecting&&stageView!=='full')showStageView('full');
    visualButton.textContent=visualSelecting?'Finish picking':'Pick on diagram';
    visualButton.setAttribute('aria-pressed',String(visualSelecting));visualButton.classList.toggle('active',visualSelecting);
    stageCards.forEach(card=>card.querySelector('.stage-pick-control').hidden=!visualSelecting);
    sizeStageSelectors();
    window.dispatchEvent(new Event('latte:stage-selection'));
  };
  const picker=document.createElement('section');picker.className='stage-picker';picker.hidden=true;picker.setAttribute('aria-label','Choose visualization steps');
  picker.innerHTML='<header><strong>Choose steps to show</strong><button type="button" aria-label="Close step picker">×</button></header><p>All steps still run. Choose the steps to display in the Selected steps tab.</p><div class="stage-picker-actions"><button type="button" data-choice="all">Select all</button><button type="button" data-choice="none">Clear</button></div><div class="stage-picker-list"></div><button type="button" class="stage-picker-apply">Show selected steps</button>';
  const list=picker.querySelector('.stage-picker-list');
  const groupNodes=[];
  function addStageChoice(card,parent){
    const label=document.createElement('label'),checkbox=document.createElement('input');checkbox.type='checkbox';checkbox.value=card.dataset.operation;checkbox.dataset.stageLeaf='true';
    const text=document.createElement('span');text.textContent=`${stageCards.indexOf(card)}. ${card.dataset.operation.replaceAll(' • ',' · ')}`;
    label.append(checkbox,text);parent.append(label);
  }
  function addChoiceGroup(label,cards,parent,path){
    const group=document.createElement('div');group.className='stage-picker-group';
    const heading=document.createElement('label');heading.className='stage-picker-group-heading';
    const checkbox=document.createElement('input');checkbox.type='checkbox';checkbox.dataset.stageGroup=path;checkbox.setAttribute('aria-label',`Select ${path}`);
    const title=document.createElement('span');title.textContent=label;heading.append(checkbox,title);
    const children=document.createElement('div');children.className='stage-picker-children';group.append(heading,children);parent.append(group);
    groupNodes.push({checkbox,children});
    return children;
  }
  function runs(items,classify){
    const result=[];
    for(const card of items){
      const label=classify(card),last=result.at(-1);
      if(label&&last?.label===label)last.cards.push(card);else result.push({label,cards:[card]});
    }
    return result;
  }
  // Match the left-bracket hierarchy: Layer → Attention / MLP / Residual,
  // and Head within Attention. Keep repeated residual runs separate.
  layerGroups.forEach(group=>{
    const cards=stageCards.filter(card=>card.closest('.layer-group')===group);
    if(!cards.length)return;
    const layer=cards[0].dataset.operation.match(/layer\s+\d+/i)?.[0];
    const label=layer?layer.replace(/^layer/i,'Layer'):(cards[0].dataset.category==='embedding'?'Embedding':'Output');
    const root=addChoiceGroup(label,cards,list,label);
    if(!layer){cards.forEach(card=>addStageChoice(card,root));return;}
    let residual=0;
    runs(cards,card=>/^(ATTENTION|MLP|RESIDUAL)/.exec(card.dataset.operation)?.[0]).forEach(section=>{
      if(!section.label){section.cards.forEach(card=>addStageChoice(card,root));return;}
      const name={ATTENTION:'Attention',MLP:'MLP',RESIDUAL:'Residual'}[section.label];
      const path=`${label} / ${name}${section.label==='RESIDUAL'?` ${residual++}`:''}`;
      const children=addChoiceGroup(name,section.cards,root,path);
      runs(section.cards,card=>card.dataset.operation.match(/head\s+\d+/i)?.[0].replace(/^head/i,'Head')).forEach(head=>{
        const parent=head.label?addChoiceGroup(head.label,head.cards,children,`${path} / ${head.label}`):children;
        head.cards.forEach(card=>addStageChoice(card,parent));
      });
    });
  });
  function syncGroupChecks(){
    groupNodes.forEach(({checkbox,children})=>{
      const leaves=[...children.querySelectorAll('input[data-stage-leaf]')],count=leaves.filter(input=>input.checked).length;
      checkbox.checked=leaves.length>0&&count===leaves.length;checkbox.indeterminate=count>0&&count<leaves.length;
    });
  }
  document.body.append(picker);
  function applyStageView(view,{preserveSelection=false}={}) {
    if(!preserveSelection)window.LattePresentation?.clear();
    restoreFocus();closeMathPanel();stageView=view;
    corpusPanel.hidden=view!=='corpus';
    document.body.classList.toggle('corpus-view',view==='corpus');
    corpusTab.classList.toggle('active',view==='corpus');corpusTab.setAttribute('aria-pressed',String(view==='corpus'));
    stepsMenu.hidden=view==='corpus';
    const selected=new Set(chosenStages);
    if(view==='selected'||view==='corpus') {
      visualSelecting=false;picker.hidden=true;
      visualButton.textContent='Pick on diagram';visualButton.setAttribute('aria-pressed','false');visualButton.classList.remove('active');
      stageCards.forEach(card=>card.querySelector('.stage-pick-control').hidden=true);
    }
    chooseButton.hidden=visualButton.hidden=view==='selected';
    const chartToggle=document.querySelector('.sampling-chart-toggle');
    if(chartToggle) {
      chartToggle.disabled=view==='corpus'||view==='selected'&&!chosenStages.some(name=>name.includes('Seeded top-k sampling'));
      chartToggle.title=chartToggle.disabled?'Select the sampling step in Full model to enable the chart':'Show or hide sampling chart';
      if(chartToggle.disabled)setSamplingChartOpen(false);
    }
    diagramChecks.forEach((checkbox,name)=>{checkbox.checked=selected.has(name);checkbox.closest('.stage-card').classList.toggle('stage-picked',checkbox.checked);});
    stageCards.forEach(card=>{card.style.display=view==='selected'?(selected.has(card.dataset.operation)?'':'none'):originalDisplays.get(card);});
    layerGroups.forEach(group=>{group.style.display=[...group.querySelectorAll('.stage-card')].some(card=>card.style.display!=='none')?'':'none';});
    fullTab.classList.toggle('active',view==='full');selectedTab.classList.toggle('active',view==='selected');
    fullTab.setAttribute('aria-pressed',String(view==='full'));selectedTab.setAttribute('aria-pressed',String(view==='selected'));
    selectedTab.disabled=!chosenStages.length;
    window.dispatchEvent(new Event('latte:stage-selection'));
    window.dispatchEvent(new Event('latte:layout'));
  }
  const showStageView=view=>{applyStageView(view);requestAnimationFrame(()=>window.LittleLlamaWorkbook.recenter?.());};
  fullTab.onclick=()=>showStageView('full');selectedTab.onclick=()=>showStageView('selected');
  corpusTab.onclick=()=>applyStageView('corpus');
  clearStagesButton.onclick=()=>{
    chosenStages=[];visualSelecting=false;picker.hidden=true;
    visualButton.textContent='Pick on diagram';visualButton.setAttribute('aria-pressed','false');visualButton.classList.remove('active');
    stageCards.forEach(card=>card.querySelector('.stage-pick-control').hidden=true);
    list.querySelectorAll('input[data-stage-leaf]').forEach(input=>input.checked=false);
    updateApply();showStageView('full');
  };
  chooseButton.onclick=()=>{list.querySelectorAll('input[data-stage-leaf]').forEach(input=>input.checked=chosenStages.includes(input.value));picker.hidden=!picker.hidden;updateApply();};
  picker.querySelector('header button').onclick=()=>{picker.hidden=true;};
  picker.querySelectorAll('[data-choice]').forEach(button=>button.onclick=()=>{list.querySelectorAll('input[data-stage-leaf]').forEach(input=>input.checked=button.dataset.choice==='all');updateApply();});
  const apply=picker.querySelector('.stage-picker-apply');
  const updateApply=()=>{syncGroupChecks();apply.disabled=!list.querySelector('input[data-stage-leaf]:checked');};
  list.addEventListener('change',event=>{
    if(event.target.matches('input[data-stage-group]')){
      event.target.closest('.stage-picker-group').querySelectorAll('input[data-stage-leaf]').forEach(input=>input.checked=event.target.checked);
    }
    updateApply();
  });updateApply();
  apply.onclick=()=>{chosenStages=[...list.querySelectorAll('input[data-stage-leaf]:checked')].map(input=>input.value);picker.hidden=true;showStageView('selected');};
  window.LittleLlamaWorkbook.stagePicking=()=>visualSelecting;
  window.LittleLlamaWorkbook.stageGroupSelection=cards=>{
    const count=cards.filter(card=>chosenStages.includes(card.dataset.operation)).length;
    return {checked:cards.length>0&&count===cards.length,indeterminate:count>0&&count<cards.length};
  };
  window.LittleLlamaWorkbook.setStageGroupSelection=(cards,checked)=>{
    const selected=new Set(chosenStages);
    cards.forEach(card=>{if(checked)selected.add(card.dataset.operation);else selected.delete(card.dataset.operation);});
    chosenStages=stageCards.filter(card=>selected.has(card.dataset.operation)).map(card=>card.dataset.operation);
    diagramChecks.forEach((input,name)=>{input.checked=selected.has(name);input.closest('.stage-card').classList.toggle('stage-picked',input.checked);});
    list.querySelectorAll('input[data-stage-leaf]').forEach(input=>input.checked=selected.has(input.value));
    selectedTab.disabled=!chosenStages.length;updateApply();
    window.dispatchEvent(new Event('latte:stage-selection'));
  };
  window.LittleLlamaWorkbook.stageSelection=()=>({stages:[...chosenStages],view:stageView});
  window.LittleLlamaWorkbook.restoreStageSelection=state=>{
    chosenStages=(state?.stages||[]).filter(name=>stageCards.some(card=>card.dataset.operation===name));
    applyStageView(state?.view==='selected'&&chosenStages.length?'selected':'full');
  };
  window.LittleLlamaWorkbook.setTrainingCorpus=corpus=>{
    corpusSource.textContent=corpus?.path?`Source: ${corpus.path}`:'Corpus source is unavailable for this export.';
    corpusText.textContent=corpus?.text||'';
  };
  window.LittleLlamaWorkbook.addSelectedStep=name=>{
    if(!chosenStages.includes(name))chosenStages.push(name);
    applyStageView('selected',{preserveSelection:true});
  };
  applyStageView('full');
  const stageStyle=document.createElement('style');
  stageStyle.textContent='.stage-view-tabs { position:fixed; left:12px; bottom:1px; height:28px; z-index:201; display:flex; gap:3px; align-items:center; font:11px system-ui; } .stage-view-tabs button { border:0; border-radius:5px; background:transparent; color:#56718f; padding:5px 10px; cursor:pointer; } .stage-view-tabs button.active { background:#e9e0f7; color:#7c3aed; } .stage-view-tabs button:disabled { opacity:.45; cursor:default; } .stage-picker { position:fixed; z-index:150; left:18px; bottom:44px; width:min(480px,calc(100vw - 36px)); max-height:calc(100vh - 150px); overflow:auto; overscroll-behavior:contain; background:#fff; border:1px solid #dce4ef; padding:16px; border-radius:12px; box-sizing:border-box; color:#405978; font:12px/1.5 system-ui; box-shadow:0 8px 35px #23304724; } .stage-picker[hidden] { display:none; } .stage-picker header { display:flex; justify-content:space-between; align-items:center; } .stage-picker button { padding:6px 10px; border:1px solid #dce4ef; border-radius:6px; background:#f7f9fc; color:#405978; cursor:pointer; } .stage-picker button:disabled { opacity:.5; cursor:default; } .stage-picker header button { font-size:18px; } .stage-picker-list { margin:12px 0; display:grid; gap:5px; } .stage-picker-list label { display:flex; gap:8px; align-items:flex-start; padding:5px; } .stage-picker-list input { margin-top:3px; } .stage-picker-actions { display:flex; gap:6px; } .stage-picker-children { margin-left:7px; padding-left:12px; border-left:1px solid #dce4ef; } .stage-picker-list .stage-picker-group-heading { font-weight:600; padding-top:8px; } .stage-picker-list .stage-picker-group-heading span { color:#684832; }';
  document.head.append(stageStyle);
  if(document.documentElement.hasAttribute('data-training-replay')){
    selectedTab.textContent='Selected blocks';stepsSummary.textContent='Choose blocks';
    chooseButton.textContent='Choose from list…';clearStagesButton.textContent='Reset selected blocks';
    clearStagesButton.title='Deselect all blocks and return to Full model';
    picker.setAttribute('aria-label','Choose visualization blocks');
    picker.querySelector('strong').textContent='Choose blocks to show';
    picker.querySelector('p').textContent='All calculations still run. Only the selected blocks are shown.';
    apply.textContent='Show selected blocks';
  }
  const visualStageStyle=document.createElement('style');
  visualStageStyle.textContent='.stage-pick-control { position:absolute; z-index:60; display:flex; align-items:center; background:transparent; border:0; padding:0; cursor:pointer; } .stage-pick-control[hidden] { display:none; } .stage-pick-control input { width:13px; height:13px; margin:0; accent-color:#879bb4; opacity:.7; cursor:pointer; }';
  document.head.append(visualStageStyle);
  const stepsMenuStyle=document.createElement('style');
  stepsMenuStyle.textContent='.steps-menu { position:relative; } .steps-menu summary { padding:5px 10px; color:#56718f; cursor:pointer; } .steps-menu-actions { position:absolute; bottom:34px; left:0; min-width:170px; padding:6px; background:#fff; border:1px solid #dce4ef; border-radius:8px; box-shadow:0 4px 18px #23304716; } .steps-menu-actions button { display:block; width:100%; text-align:left; white-space:nowrap; } .steps-menu-actions button[hidden] { display:none; } .steps-menu-actions button:hover { background:#f0f3f8; }';
  document.head.append(stepsMenuStyle);
  const hiddenInputsStyle=document.createElement('style');
  hiddenInputsStyle.textContent='.hidden-input-badge { display:block; color:#70819a; font:11px/1.4 system-ui; padding:4px 0; } .input-matrix-card .hidden-input-action { display:block; font:11px/1.4 system-ui; color:#56718f; text-decoration:underline; padding:3px 0; }';
  document.head.append(hiddenInputsStyle);
  const corpusStyle=document.createElement('style');
  corpusStyle.textContent='.training-corpus-panel { position:fixed; z-index:125; inset:76px 0 30px; background:#f4f6fa; overflow:auto; overscroll-behavior:contain; padding:28px max(24px,calc((100vw - 920px)/2)); box-sizing:border-box; color:#405978; font:14px/1.7 system-ui; } .training-corpus-panel[hidden] { display:none; } .training-corpus-panel h2 { margin:0 0 12px; font-size:20px; } .training-corpus-panel p { color:#70819a; font-size:11px; overflow-wrap:anywhere; } .training-corpus-panel pre { white-space:pre-wrap; overflow-wrap:anywhere; font:14px/1.8 system-ui; }';
  document.head.append(corpusStyle);
  const corpusIsolationStyle=document.createElement('style');
  corpusIsolationStyle.textContent='body.corpus-view .worksheet, body.corpus-view .annotation-overlay, body.corpus-view .bracket-fade, body.corpus-view .dependency-overlay, body.corpus-view .edge-connections { visibility:hidden !important; pointer-events:none !important; }';
  document.head.append(corpusIsolationStyle);
})();
