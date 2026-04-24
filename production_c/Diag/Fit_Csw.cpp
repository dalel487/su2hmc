#include	<cmath>
#include <cstdlib>
#include	<cstdio>
#include <vector>

#include	"TAxis.h"
#include	"TCanvas.h"
#include	"TF1.h"
#include	"TGraph.h"
#include	"TGraphErrors.h"
#include	"TLine.h"
#include "TROOT.h"
#include "TStyle.h"

int main(int argc, char *argv[]){


	//	gROOT->SetStyle("BELLE2");
	gStyle->SetEndErrorSize(15);//gStyle->SetLegendFillColor(0);
	gStyle->SetLegendBorderSize(0);gStyle->SetTitleFont(42,"");
	gStyle->SetTitleSize(0.3,"t"); gStyle->SetTitleFontSize(0.05);
	gStyle->SetExponentOffset(-0.08,0,"y");gROOT->ForceStyle();

	//Array for A values
	double *rat_a[2], *diff_a[2];
	for(unsigned short i=0;i<2;i++){
		rat_a[i]=(double*)malloc(sizeof(double)*(argc-1));
		diff_a[i]=(double*)malloc(sizeof(double)*(argc-1));
	}
	double *c_sw=(double*)malloc(sizeof(double)*(argc-1));
	const char *name_eps = "#varepsilon";

	for(unsigned int i=1; i<argc;i++){
		c_sw[i-1]=strtof(argv[i],NULL);
		char infile_name[64];
		sprintf(infile_name,"Force_Action_Check_%1.2f",c_sw[i-1]);
		FILE *infile=fopen(infile_name,"r");

		std::vector<float> eps, rat, norm_diff;
		int skipped=0; char line[128];
		while (fgets(line,sizeof(line),infile)) {
			// Skip blank lines
			if (line[0]=='\n') continue;

			// Skip the first two non-blank lines (|dSdpi| and header)
			if (skipped < 2) {
				skipped++;
				continue;
			}

			float col1, col4, col5;
			if (sscanf(line,"%e %*e %*e %f %e",&col1,&col4,&col5)!=3) continue;

			eps.push_back(col1);
			rat.push_back(col4);
			norm_diff.push_back(col5);
		}
		fclose(infile);

		if(eps.empty()){
			fprintf(stderr,"Error for %s. Epsilon array is empty.\nSkipping\n\n",argv[i]);
			continue;
		}
		//Fit ratio and extract linear term
		char csw_name[16];
		sprintf(csw_name,"C_sw_%s",argv[i]);
		TCanvas *c1 = new TCanvas(csw_name,csw_name,2970,2100);
		//	c1->SetMargin(0.2,0.2,0.2,0);
		c1->Divide(2,1,0.03);

		TGraph *g_rat=new TGraph(eps.size(),eps.data(),rat.data());
		g_rat->SetBit(kCanDelete);
		g_rat->SetMarkerColor(kBlue-3); g_rat->SetMarkerStyle(24);
		TF1 *rat_fit = new TF1("Fit","1+[2]+[0]*x+[1]*x*x",1e-9,0.11);
		rat_fit->SetBit(kCanDelete);
		rat_fit->SetLineColor(kOrange+7); rat_fit->SetFillColorAlpha(kOrange+7,0);
		rat_fit->SetLineWidth(1);
		printf("Fitting ratio for %s\n",argv[i]);
		g_rat->Fit(rat_fit,"GME");

		char rat_title[64];
		sprintf(rat_title,"Ratio c_{sw}=%s;%s;#frac{dS_{Num}}{dS_{Ana}}",argv[i],name_eps);
		g_rat->SetTitle(rat_title);
		c1->cd(1);
//		gPad->SetLogx();
		g_rat->Draw("AP");
		rat_a[0][i-1]=(float)rat_fit->GetParameter(0);
		rat_a[1][i-1]=(float)rat_fit->GetParError(0);

		//Fit diff and extract linear term
		TGraph *g_diff=new TGraph(eps.size(),eps.data(),norm_diff.data());
		g_diff->SetBit(kCanDelete);
		g_diff->SetMarkerColor(kBlue-3); g_diff->SetMarkerStyle(24);
		TF1 *diff_fit = new TF1("Fit","[0]/x+[1]*x+[2]",1e-9,0.011);
		diff_fit->SetBit(kCanDelete);
		diff_fit->SetLineColor(kOrange+7); diff_fit->SetFillColorAlpha(kOrange+7,0);
		diff_fit->SetLineWidth(1);
		printf("Fitting diff for %s\n",argv[i]);
		g_diff->Fit(diff_fit,"GME");

		char diff_title[64];
		sprintf(diff_title,"Diff c_{sw}=%s;%s;#frac{dS_{Num}-dS_{Ana}}{#varepsilon^{2}}",argv[i],name_eps);
		g_diff->SetTitle(diff_title);
		c1->cd(2);
//		gPad->SetLogx();
		g_diff->Draw("AP");
		diff_a[0][i-1]=(float)diff_fit->GetParameter(0);
		diff_a[1][i-1]=(float)diff_fit->GetParError(0);
		char outname[32];
		sprintf(outname,"%s.svg",csw_name);
		c1->Print(outname);
		delete c1;
	}
	const char *name_rat = "Ratio_A_Fit"; const char *name_diff = "Diff_A_Fit";
	const char *quad="[0]+[1]*x+[2]*x*x";
	TCanvas *c1 = new TCanvas(name_rat,name_rat,2970,2100);
	c1->SetTopMargin(0.1);	c1->SetRightMargin(0.1);
	c1->Divide(2,1);

	TGraphErrors *g_rat = new TGraphErrors(argc-1,c_sw,rat_a[0],NULL,rat_a[1]);
	g_rat->SetBit(kCanDelete);
	g_rat->SetMarkerColor(kBlue-3); g_rat->SetMarkerStyle(24);
	g_rat->SetLineColor(kBlue-3);
	TF1 *rat_fit = new TF1(name_rat,quad, c_sw[0],c_sw[argc-2]);
	rat_fit->SetBit(kCanDelete);
	rat_fit->SetLineColor(kOrange+7); rat_fit->SetFillColorAlpha(kOrange+7,0);
	rat_fit->SetLineWidth(1);
	printf("Fitting ratio linear terms\n");
	g_rat->Fit(rat_fit,"GME");
	char rat_title[64];
	sprintf(rat_title,"%s;c_{sw};Linear Coefficient",name_rat);
	g_rat->SetTitle(rat_title);
	c1->cd(1);
	g_rat->Draw("AP");

	TGraphErrors *g_diff = new TGraphErrors(argc-1,c_sw,diff_a[0],NULL,diff_a[1]);
	g_diff->SetBit(kCanDelete);
	g_diff->SetMarkerColor(kBlue-3); g_diff->SetMarkerStyle(24);
	g_diff->SetLineColor(kBlue-3);
	TF1 *diff_fit = new TF1(name_diff,quad, c_sw[0],c_sw[argc-2]);
	diff_fit->SetBit(kCanDelete);
	diff_fit->SetLineColor(kOrange+7); diff_fit->SetFillColorAlpha(kOrange+7,0);
	diff_fit->SetLineWidth(1);
	printf("Fitting diff linear terms\n");
	g_diff->Fit(diff_fit,"GME");
	char diff_title[64];
	sprintf(diff_title,"%s;c_{sw};Linear Coefficient",name_diff);
	g_diff->SetTitle(diff_title);
	c1->cd(2);
	g_diff->Draw("AP");

	c1->Print("A_fits.svg");
	printf("Ratio fit %ex^2+%ex+%e\n",rat_fit->GetParameter(2),rat_fit->GetParameter(1),rat_fit->GetParameter(0));
	printf("Diff fit %ex^2+%ex+%e\n",diff_fit->GetParameter(2),diff_fit->GetParameter(1),diff_fit->GetParameter(0));
	delete c1;

	for(unsigned short i=0;i<2;i++){
		free(rat_a[i]); free(diff_a[i]);
	}
	free(c_sw);
	return 0;
}
