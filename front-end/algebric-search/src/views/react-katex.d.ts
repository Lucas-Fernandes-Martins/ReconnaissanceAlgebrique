// src/react-katex.d.ts
declare module 'react-katex' {
    import * as React from 'react';
  
    export interface KatexProps {
      children?: string;
      math?: string;
      errorColor?: string;
      renderError?: (error: Error) => React.ReactNode;
      settings?: object;
    }
  
    export class InlineMath extends React.Component<KatexProps> {}
    export class BlockMath extends React.Component<KatexProps> {}
  }
  