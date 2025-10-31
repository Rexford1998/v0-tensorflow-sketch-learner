import SketchLearner from "@/components/sketch-learner"

export default function Home() {
  return (
    <main className="min-h-screen bg-background">
      <div className="container mx-auto py-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-2">AI Sketch Learner</h1>
          <p className="text-muted-foreground">Draw, label, train, and predict with TensorFlow.js and Firebase</p>
        </div>
        <SketchLearner />
      </div>
    </main>
  )
}
